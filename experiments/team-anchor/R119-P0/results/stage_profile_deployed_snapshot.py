#!/usr/bin/env python3
"""Attribute actual LayerSkip draft/verify/cache/selection stages on the R119 GPU."""
import argparse, copy, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.generation.candidate_generator import EarlyExitCandidateGenerator
import transformers.generation.utils as generation_utils


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--repeats',type=int,default=5); ap.add_argument('--tokens',type=int,default=8); ap.add_argument('--docs',type=int,default=6); args=ap.parse_args()
    prep=json.loads((args.root/'prepared.json').read_text()); prompts=json.loads((args.root/'prompts.json').read_text())
    model=AutoModelForCausalLM.from_pretrained(prep['snapshot'],torch_dtype=torch.float16,use_safetensors=True,local_files_only=True).cuda().eval()
    tok=AutoTokenizer.from_pretrained(prep['snapshot'],local_files_only=True)
    rows=[r for r in prompts if r['split']=='test'][:args.docs]
    state={'in_draft':False,'draft_host':[],'draft_events':[],'verify_host':[],'verify_events':[],'cache_host':[],'cache_events':[],'sample_host':[],'sample_events':[],'crop_host':[],'candidates':[],'matches':[]}
    def ev():
        e=torch.cuda.Event(enable_timing=True); e.record(); return e
    orig_get=EarlyExitCandidateGenerator.get_candidates
    def get_candidates(self,input_ids):
        t=time.perf_counter(); state['in_draft']=True
        try: result=orig_get(self,input_ids)
        finally: state['in_draft']=False
        state['draft_host'].append(time.perf_counter()-t)
        state['candidates'].append(int(result[0].shape[1]-input_ids.shape[1]))
        return result
    orig_cache=DynamicCache.update
    def cache_update(self,*a,**kw):
        t=time.perf_counter(); start=ev()
        try:return orig_cache(self,*a,**kw)
        finally:state['cache_host'].append(time.perf_counter()-t);state['cache_events'].append((start,ev()))
    orig_update=EarlyExitCandidateGenerator.update_candidate_strategy
    def update_strategy(self,input_ids,scores,num_matches):
        state['matches'].append(num_matches.detach().clone() if isinstance(num_matches,torch.Tensor) else num_matches)
        return orig_update(self,input_ids,scores,num_matches)
    orig_argmax=torch.Tensor.argmax
    def argmax(self,*a,**kw):
        t=time.perf_counter();start=ev()
        try:return orig_argmax(self,*a,**kw)
        finally:state['sample_host'].append(time.perf_counter()-t);state['sample_events'].append((start,ev()))
    orig_torch_argmax=torch.argmax
    def torch_argmax(*a,**kw):
        t=time.perf_counter();start=ev()
        try:return orig_torch_argmax(*a,**kw)
        finally:state['sample_host'].append(time.perf_counter()-t);state['sample_events'].append((start,ev()))
    orig_crop=generation_utils._crop_past_key_values
    def crop(*a,**kw):
        t=time.perf_counter()
        try:return orig_crop(*a,**kw)
        finally:state['crop_host'].append(time.perf_counter()-t)
    def pre(module,inputs):
        item=(state['in_draft'],time.perf_counter(),ev()); state.setdefault('_forward',[]).append(item)
    def post(module,inputs,out):
        draft,t,start=state['_forward'].pop(); elapsed=time.perf_counter()-t; stop=ev()
        k='draft' if draft else 'verify'; state[k+'_host'].append(elapsed);state[k+'_events'].append((start,stop))
    handles=[model.register_forward_pre_hook(pre),model.register_forward_hook(post)]
    EarlyExitCandidateGenerator.get_candidates=get_candidates; EarlyExitCandidateGenerator.update_candidate_strategy=update_strategy; DynamicCache.update=cache_update; torch.Tensor.argmax=argmax; torch.argmax=torch_argmax; generation_utils._crop_past_key_values=crop
    records=[]
    try:
      for length in (64,256,512):
       for arm in ('AR','LayerSkip'):
        cfg=copy.deepcopy(model.generation_config);cfg.max_new_tokens=args.tokens;cfg.do_sample=False;cfg.num_beams=1;cfg.pad_token_id=tok.eos_token_id
        if arm=='LayerSkip':cfg.assistant_early_exit=4;cfg.num_assistant_tokens=2;cfg.num_assistant_tokens_schedule='constant';cfg.assistant_confidence_threshold=0.0
        model.generation_config=copy.deepcopy(cfg) if arm=='LayerSkip' else model.generation_config
        if arm=='LayerSkip':model.generation_config.assistant_early_exit=None
        for row in rows:
         x=tok(row['text'],return_tensors='pt',truncation=True,max_length=512)['input_ids'].cuda()
         if x.shape[1]<length:x=x.repeat(1,(length+x.shape[1]-1)//x.shape[1])
         x=x[:,:length]
         model.generate(input_ids=x,max_new_tokens=2,do_sample=False,pad_token_id=tok.eos_token_id)
         for rep in range(args.repeats):
          for k in ('draft_host','draft_events','verify_host','verify_events','cache_host','cache_events','sample_host','sample_events','crop_host','candidates','matches'):
           state[k].clear()
          torch.cuda.synchronize();t=time.perf_counter();out=model.generate(input_ids=x,generation_config=cfg);torch.cuda.synchronize();wall=time.perf_counter()-t
          def event_sum(key):return sum(a.elapsed_time(b)/1000 for a,b in state[key])
          # Snapshot mutable instrumentation buffers before the next repetition clears them.
          cand=list(state['candidates']); match=list(state['matches'])
          records.append(dict(length=length,arm=arm,doc_id=row['id'],rep=rep,wall_seconds=wall,generated_tokens=int(out.shape[1]-x.shape[1]),draft_get_candidates_host_seconds=sum(state['draft_host']),draft_forward_cuda_event_seconds=event_sum('draft_events'),verify_forward_host_seconds=sum(state['verify_host']),verify_forward_cuda_event_seconds=event_sum('verify_events'),kv_update_host_seconds=sum(state['cache_host']),kv_update_cuda_event_seconds=event_sum('cache_events'),selection_argmax_host_seconds=sum(state['sample_host']),selection_argmax_cuda_event_seconds=event_sum('sample_events'),cache_crop_host_seconds=sum(state['crop_host']),candidate_lengths=cand,accepted_draft_matches=[int(m.item()) if isinstance(m,torch.Tensor) else int(m) for m in match],rollback_tokens=[max(0,c-(int(m) if not isinstance(m,torch.Tensor) else int(m.item()))) for c,m in zip(cand,match)]))
    finally:
      EarlyExitCandidateGenerator.get_candidates=orig_get;EarlyExitCandidateGenerator.update_candidate_strategy=orig_update;DynamicCache.update=orig_cache;torch.Tensor.argmax=orig_argmax;torch.argmax=orig_torch_argmax;generation_utils._crop_past_key_values=orig_crop
      for h in handles:h.remove()
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    print(json.dumps({'rows':len(records),'gpu':torch.cuda.get_device_name(),'repeats':args.repeats,'docs':len(rows),'out':str(args.out)},indent=2))

if __name__=='__main__':main()
