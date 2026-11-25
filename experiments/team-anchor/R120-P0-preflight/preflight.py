"""R120: cached greedy oracle diagnostic; NOT a deployable draft or benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest


def verify(prompt, n, k, backend, tape=None):
    """Cache contains committed prefix except pending last token. No EOS stopping."""
    if not prompt or n < 1 or k < 0 or (k and (tape is None or len(tape) != n)):
        raise ValueError('invalid prompt, length, or tape')
    cache = backend.prefill(prompt[:-1])
    pending = prompt[-1]
    out, blocks = [], []
    while len(out) < n:
        start = len(prompt) + len(out) - 1
        assert backend.length(cache) == start
        m = min(k, n - len(out) - 1)
        draft = list(tape[len(out):len(out)+m]) if m else []
        ids = [pending] + draft
        pred, cache = backend.forward(ids, cache)
        assert len(pred) == m + 1
        assert backend.length(cache) == start + m + 1
        a = 0
        while a < m and pred[a] == draft[a]:
            a += 1
        emitted = draft[:a] + [pred[a]]
        keep = start + 1 + a
        cache = backend.crop(cache, keep)
        assert backend.length(cache) == keep
        blocks.append(dict(cache_before=start, cache_after=keep, draft=draft,
                           predictions=pred, accepted=a, emitted=emitted))
        out.extend(emitted)
        pending = emitted[-1]
    assert len(out) == n
    return dict(output_ids=out, blocks=blocks)


class Toy:
    """History-sensitive deterministic backend, intentionally mutable cache."""
    def prefill(self, prefix):
        return list(prefix)
    def length(self, cache):
        return len(cache)
    def forward(self, ids, cache):
        pred = []
        for x in ids:
            cache.append(x)
            pred.append((sum((i+1)*v for i,v in enumerate(cache))+len(cache)) % 31)
        return pred, cache
    def crop(self, cache, length):
        del cache[length:]
        return cache


class Tests(unittest.TestCase):
    def test_exhaustive_tail_and_rejection(self):
        for prompt in ([2], [3,7,9], [0,3,2,10]):
            for n in range(1,19):
                ar = verify(prompt,n,0,Toy())['output_ids']
                for k in (1,2,4,8,16):
                    tapes = [ar] + [ar[:j]+[(ar[j]+1)%31]+ar[j+1:] for j in range(n)]
                    for tape in tapes:
                        r = verify(prompt,n,k,Toy(),tape)
                        self.assertEqual(r['output_ids'],ar)
                        self.assertEqual([v for b in r['blocks'] for v in b['emitted']],ar)
                        for b in r['blocks']:
                            self.assertEqual(b['cache_after'],b['cache_before']+1+b['accepted'])
    def test_invalid(self):
        for args in (([],3,0,None),([1],0,0,None),([1],3,4,[1])):
            with self.assertRaises(ValueError):
                verify(args[0],args[1],args[2],Toy(),args[3])


class HFBackend:
    def __init__(self, model, torch, cache_cls):
        self.model, self.torch, self.cache_cls = model, torch, cache_cls
    def prefill(self, prefix):
        cache = self.cache_cls()
        if prefix:
            _, cache = self.forward(prefix,cache)
        return cache
    def length(self, cache):
        return int(cache.get_seq_length())
    def forward(self, ids, cache):
        t = self.torch
        start = self.length(cache)
        inp = t.tensor([ids],device=self.model.device,dtype=t.long)
        pos = t.arange(start,start+len(ids),device=inp.device)
        mask = t.ones((1,start+len(ids)),device=inp.device,dtype=t.long)
        with t.inference_mode():
            r = self.model(input_ids=inp, attention_mask=mask, position_ids=pos.unsqueeze(0),
                           cache_position=pos,past_key_values=cache,use_cache=True,return_dict=True)
            pred = r.logits[0].argmax(-1).tolist()
        return pred,r.past_key_values
    def crop(self, cache, length):
        cache.crop(length)
        return cache


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--model',type=Path)
    p.add_argument('--prompts',type=Path,help='JSON array of two distinct token-ID arrays of lengths 64 and 256')
    p.add_argument('--out',type=Path)
    p.add_argument('--approve-preflight',action='store_true')
    args = p.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not (args.approve_preflight and args.model and args.prompts and args.out):
        p.error('requires --approve-preflight --model --prompts --out; no formal benchmark')
    if args.out.exists():
        p.error('output already exists; preserve prior records')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, DynamicCache, GenerationConfig
    if transformers.__version__ != '4.48.3' or not torch.cuda.is_available():
        p.error('requires Transformers 4.48.3 and CUDA')
    weights = args.model/'model.safetensors'
    expected='ba9a27bc1fadc74d7ca576aa618e464aa006b1c56b4a940c668c980a302708ee'
    if file_hash(weights) != expected or list(args.model.glob('*.safetensors')) != [weights]:
        p.error('unexpected checkpoint files or hash')
    prompts=json.loads(args.prompts.read_text())
    if not isinstance(prompts,list) or sorted(map(len,prompts)) != [64,256]:
        p.error('need two prompts, lengths 64 and 256')
    model=AutoModelForCausalLM.from_pretrained(str(args.model),local_files_only=True,
            torch_dtype=torch.float16,attn_implementation='eager',use_safetensors=True).to('cuda').eval()
    backend=HFBackend(model,torch,DynamicCache)
    records=[]
    report=dict(scope='minimal interface preflight only; no speed claims',torch=torch.__version__,
                transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
                source_sha256=file_hash(Path(__file__)),model_files={f.name:file_hash(f) for f in args.model.iterdir()
                if f.is_file() and (f.suffix=='.json' or f==weights)},records=records)
    try:
        for prompt in prompts:
            if any(type(x) is not int or x<0 or x>=model.config.vocab_size for x in prompt):
                raise ValueError('invalid token ID')
            ar=verify(prompt,16,0,backend)
            with torch.inference_mode():
                ids=torch.tensor([prompt],device=model.device)
                gen=model.generate(ids,attention_mask=torch.ones_like(ids),
                         generation_config=GenerationConfig(do_sample=False,max_new_tokens=16,
                             eos_token_id=None,pad_token_id=0,use_cache=True))
            reference=gen[0,len(prompt):].tolist()
            if len(reference)!=16:
                raise RuntimeError('reference generation length differs; stop')
            row=dict(prompt_ids=prompt,reference_ids=reference,manual_ar=ar,arms=[])
            records.append(row)
            if ar['output_ids']!=reference:
                raise RuntimeError('manual AR differs from generate; stop')
            for k in (1,4,8):
                for corrupt in (False,True):
                    tape=reference.copy()
                    if corrupt:
                        tape[0]=(tape[0]+1)%model.config.vocab_size
                    torch.cuda.reset_peak_memory_stats()
                    r=verify(prompt,16,k,backend,tape)
                    r.update(k=k,corrupt=corrupt,match=r['output_ids']==reference,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    row['arms'].append(r)
                    if not r['match']:
                        raise RuntimeError('block output differs from AR; stop')
                    if corrupt and r['blocks'][0]['accepted']!=0:
                        raise RuntimeError('wrong first candidate was not rejected')
                    if not corrupt and any(b['accepted']!=len(b['draft']) for b in r['blocks']):
                        raise RuntimeError('oracle candidate rejection; investigate precision')
        report['passed']=True
    except Exception as e:
        report.update(passed=False,error=repr(e))
        raise
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':
    sys.exit(main())
