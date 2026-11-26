"""R120-P2 retry: fixed-length generation protocol, reuse P2 diagnostic logic."""
import argparse
import copy
import json
from pathlib import Path
import unittest
from types import SimpleNamespace


def fixed_generate(model, x, mask, cfg, **kwargs):
    if cfg.eos_token_id is not None or cfg.do_sample or cfg.max_new_tokens is None:
        raise ValueError('expected fixed-length greedy configuration without EOS')
    original = model.generation_config
    assistant_cfg = copy.deepcopy(cfg)
    assistant_cfg.assistant_early_exit = None
    # HF 4.48.3 copies missing special tokens from model.generation_config.
    # Disable EOS in BOTH configs while keeping assistant window controls.
    model.generation_config = assistant_cfg
    try:
        return model.generate(x, attention_mask=mask, generation_config=cfg, **kwargs)
    finally:
        model.generation_config = original


class Tests(unittest.TestCase):
    def test_eos_fallback_and_assistant(self):
        class Fake:
            generation_config = SimpleNamespace(eos_token_id=[128001,128009])
            def generate(self, x, attention_mask, generation_config, **kwargs):
                effective = generation_config.eos_token_id
                if effective is None:
                    effective = self.generation_config.eos_token_id
                self_test.assertIsNone(effective)
                self_test.assertIsNone(self.generation_config.assistant_early_exit)
                self_test.assertEqual(self.generation_config.num_assistant_tokens, 2)
                return [128001]*generation_config.max_new_tokens
        self_test = self
        m = Fake(); original = m.generation_config
        for e in (None,3,4):
            cfg = SimpleNamespace(eos_token_id=None, do_sample=False, max_new_tokens=64,
                                  assistant_early_exit=e, num_assistant_tokens=2)
            self.assertEqual(len(fixed_generate(m, None, None, cfg)),64)
            self.assertIs(m.generation_config,original)
            self.assertEqual(cfg.assistant_early_exit,e)
    def test_restore_on_failure(self):
        class Fake:
            generation_config = SimpleNamespace(eos_token_id=2)
            def generate(self,*args,**kwargs):
                raise RuntimeError('injected')
        m=Fake(); original=m.generation_config
        cfg=SimpleNamespace(eos_token_id=None,do_sample=False,max_new_tokens=64)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            fixed_generate(m,None,None,cfg)
        self.assertIs(m.generation_config,original)
        cfg.eos_token_id=2
        with self.assertRaises(ValueError):
            fixed_generate(m,None,None,cfg)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--model',type=Path); p.add_argument('--out',type=Path); args=p.parse_args()
    if args.self_test:
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if not(args.model and args.out): p.error('need --model --out')
    args.out.mkdir(parents=True,exist_ok=False)
    import torch, transformers
    from transformers import AutoModelForCausalLM, DynamicCache, GenerationConfig
    from diagnose import EVENTS, KS, N, chunks, canonical, Backend, first_diff
    from preflight import file_hash
    if transformers.__version__!='4.48.3' or not torch.cuda.is_available(): raise RuntimeError('environment mismatch')
    previous=Path(__file__).resolve().parents[2]/'experiments/team-anchor/R120-P1'
    manifest=json.loads((previous/'manifest.json').read_text()); refs=json.loads((previous/'references.json').read_text())
    for name,h in dict(manifest['model_files'],**{'model.safetensors':manifest['weight_sha256']}).items():
        if file_hash(args.model/name)!=h: raise RuntimeError('model file mismatch '+name)
    if list(args.model.glob('*.safetensors')) != [args.model/'model.safetensors']: raise RuntimeError('unexpected weights')
    prompts={r['id']:r['input_ids'] for r in manifest['docs'] if r['id'] in EVENTS}
    if set(prompts)!=set(EVENTS) or any(len(refs[i])!=N for i in EVENTS): raise RuntimeError('unexpected data')
    report=dict(scope='three known failure documents; diagnostic only; FP32 separate numerical target',source_sha256=file_hash(Path(__file__)),diagnose_sha256=file_hash(Path(__file__).with_name('diagnose.py')),preflight_sha256=file_hash(Path(__file__).with_name('preflight.py')),manifest_sha256=file_hash(previous/'manifest.json'),references_sha256=file_hash(previous/'references.json'),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),runs=[],protocol_failures=[])
    def save(): (args.out/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def scores(v,ids):
        v=v.float(); val,ix=v.topk(5); return dict(top_ids=ix.tolist(),top_logits=val.tolist(),margin=float(val[0]-val[1]),tracked={str(i):float(v[i]) for i in ids})
    try:
        for dtype in (torch.float16,torch.float32):
            if dtype==torch.float32:
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
            model=AutoModelForCausalLM.from_pretrained(str(args.model),local_files_only=True,torch_dtype=dtype,attn_implementation='eager',use_safetensors=True).to('cuda').eval()
            backend=Backend(model,torch,DynamicCache)
            with torch.inference_mode():
                for rid,prompt in prompts.items():
                    row=dict(dtype=str(dtype),id=rid,default_eos_token_id=model.generation_config.eos_token_id,stage='ar',events=[],canonical=[],sd=[])
                    report['runs'].append(row); save()
                    tape=refs[rid]; x=torch.tensor([prompt],device=model.device); mask=torch.ones_like(x)
                    cfg=GenerationConfig(do_sample=False,max_new_tokens=N,eos_token_id=None,pad_token_id=0,use_cache=True)
                    g=fixed_generate(model,x,mask,cfg,return_dict_in_generate=True,output_scores=True)
                    ar=g.sequences[0,len(prompt):].tolist()
                    row.update(ar_ids=ar,ar_length=len(ar),score_steps=len(g.scores),first_diff_from_p1=first_diff(ar,tape))
                    if len(ar)!=N or len(g.scores)!=N:
                        row['stage']='ar_length_failed'; report['protocol_failures'].append([str(dtype),rid,'ar_length']); del g; save(); continue
                    row['stage']='probes'; save()
                    for j,k,other in EVENTS[rid]:
                        wanted=[tape[j],other]; event=dict(position=j+1,k=k,expected=tape[j],other=other,probes=[])
                        if ar[:j]==tape[:j]: event['generate']=scores(g.scores[j][0],wanted)
                        stream=prompt+tape
                        for style,width,last in [('split',k,False),('canonical',k,False),('full',0,False),('full',0,True)]:
                            c=DynamicCache(); plan=chunks(len(prompt),N,j,width,style)
                            for a,b in plan: v,c=backend.logits(stream[a:b],c,last=last)
                            index=0 if last else len(prompt)+j-1-plan[-1][0]
                            event['probes'].append(dict(style=style,last_only=last,intervals=plan,**scores(v[index],wanted)))
                            del c,v
                        row['events'].append(event)
                    del g
                    row['stage']='canonical'; save()
                    for k in KS:
                        z=canonical(prompt,N,k,backend,ar)
                        z.update(k=k,first_diff=first_diff(z['output_ids'],ar),rejected=sum(len(b['draft'])-b['accepted']>0 for b in z['blocks']))
                        row['canonical'].append(z)
                    row['stage']='sd'; save()
                    for e in (3,4):
                        c=copy.deepcopy(cfg); c.assistant_early_exit=e; c.num_assistant_tokens=2; c.num_assistant_tokens_schedule='constant'; c.assistant_confidence_threshold=0.0
                        s=fixed_generate(model,x,mask,c)[0,len(prompt):].tolist()
                        row['sd'].append(dict(exit=e,k=2,output_ids=s,length=len(s),first_diff=first_diff(s,ar)))
                        if len(s)!=N: report['protocol_failures'].append([str(dtype),rid,'sd_length',e])
                    row['stage']='finished'; save()
            del backend,model
            torch.cuda.empty_cache()
        report['completed']=not report['protocol_failures']
    except Exception as e:
        report.update(completed=False,error=repr(e)); raise
    finally: save()
    return 0 if report['completed'] else 1


if __name__=='__main__': raise SystemExit(main())
