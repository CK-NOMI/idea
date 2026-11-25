"""Paired greedy AR and self-speculation on the same LayerSkip checkpoint."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version


def summarize(rows):
    return dict(runs=len(rows), equal=sum(r["equal"] for r in rows),
                tokens=sum(r["tokens"] for r in rows),
                seconds=sum(r["seconds"] for r in rows))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--exits", type=int, nargs="+", default=[3, 8])
    p.add_argument("--drafts", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--seed", type=int, default=118)
    args = p.parse_args()
    started = time.perf_counter()
    cuda = args.device == "cuda"
    if cuda:
        assert torch.cuda.is_available(), "CUDA unavailable"
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    root = args.root
    prep = json.loads((root / "prepared.json").read_text())
    raw = (root / "prompts.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prep["prompt_sha256"]
    prompts = json.loads(raw)
    dev = args.device
    model = AutoModelForCausalLM.from_pretrained(
        prep["snapshot"], torch_dtype=torch.float16 if cuda else torch.float32,
        use_safetensors=True, local_files_only=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(prep["snapshot"], local_files_only=True)
    depth = model.config.num_hidden_layers
    modes = [(0, 0)] + [(e, b) for e in args.exits for b in args.drafts]
    assert all(0 < e < depth for e in args.exits)
    inputs = []
    count = {}
    for row in prompts:
        key = (row["split"], row["client"])
        count[key] = count.get(key, 0) + 1
        if count[key] <= args.per_class:
            inputs.append((row, tok(row["text"], return_tensors="pt",
                                    truncation=True, max_length=args.context).to(dev)))

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def generate(ids, arm):
        e, b = arm
        cfg = copy.deepcopy(model.generation_config)
        cfg.max_new_tokens = args.tokens
        cfg.do_sample = False
        cfg.num_beams = 1
        cfg.pad_token_id = tok.eos_token_id
        cfg.assistant_early_exit = e or None
        if e:
            cfg.num_assistant_tokens = b
            cfg.num_assistant_tokens_schedule = "constant"
            cfg.assistant_confidence_threshold = 0.0
        original = model.generation_config
        if e:
            # Transformers 4.48.3 reads these controls from the assistant's model
            # config. In self-speculation the assistant is this same model.
            model.generation_config = copy.deepcopy(cfg)
            model.generation_config.assistant_early_exit = None
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            generated = model.generate(**ids, generation_config=cfg)
            sync()
        finally:
            model.generation_config = original
        elapsed = time.perf_counter() - t
        out = generated[0, ids.input_ids.shape[-1]:].tolist()
        assert out
        return dict(seconds=elapsed, tokens=len(out), ids=out,
                    peak_bytes=torch.cuda.max_memory_allocated() if cuda else None)

    for arm in modes:
        generate(inputs[0][1], arm)
    observations = []
    raw_output = args.out.with_suffix(".jsonl")
    with raw_output.open("w") as f:
        for i, (row, ids) in enumerate(inputs):
            for rep in range(args.repeats):
                shuffled = modes.copy()
                random.Random(args.seed + 100 * i + rep).shuffle(shuffled)
                z = {m: generate(ids, m) for m in shuffled}
                for arm in modes:
                    sample = z[arm]
                    sample.update(id=row["id"], split=row["split"], client=row["client"],
                                  context=ids.input_ids.shape[-1], rep=rep, arm=arm,
                                  equal=sample["ids"] == z[(0, 0)]["ids"])
                    observations.append(sample)
                    f.write(json.dumps(sample) + "\n")
            f.flush()
            if (i + 1) % 4 == 0:
                print("Completed", i + 1, "of", len(inputs), "paired prompts", flush=True)
        print("Greedy paired runs", len(observations), flush=True)

    def select(arm, split):
        return summarize([r for r in observations if r["arm"] == arm and r["split"] == split])

    cal = {m: select(m, "calibration") for m in modes}
    test = {m: select(m, "test") for m in modes}
    valid = [m for m in modes[1:] if cal[m]["equal"] == cal[m]["runs"]]
    best = min(valid, key=lambda m: cal[m]["seconds"] / cal[m]["tokens"]) if valid else None
    locked = {}
    if best is not None:
        ref = test[(0, 0)]
        selected = test[best]
        rng = random.Random(args.seed)
        by_id = {}
        for row in observations:
            if row["split"] == "test" and row["arm"] in ((0, 0), best):
                by_id.setdefault(row["id"], {}).setdefault(row["arm"], []).append(row)
        docs = [(summarize(arms[(0, 0)]), summarize(arms[best])) for arms in by_id.values()]
        def ratio(items):
            ar, sd = zip(*items)
            ar_rate = sum(x["tokens"] for x in ar) / sum(x["seconds"] for x in ar)
            sd_rate = sum(x["tokens"] for x in sd) / sum(x["seconds"] for x in sd)
            return sd_rate / ar_rate
        samples = sorted(ratio(rng.choices(docs, k=len(docs))) for _ in range(3000))
        locked = dict(best=best, ref=ref, test=selected, speedup=ratio(docs),
                      paired_document_bootstrap_95=[samples[75], samples[2924]],
                      exact_on_test=selected["equal"] == selected["runs"])

    profiles = []
    if cuda:
        launches = []
        def before(_, positional, keywords):
            t = torch.cuda.Event(enable_timing=True)
            t.record()
            launches.append([model.config.num_hidden_layers, keywords["input_ids"].shape[-1],
                             t, None])
        def after(*_):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            launches[-1][3] = event
        pre_hook = model.register_forward_pre_hook(before, with_kwargs=True)
        post_hook = model.register_forward_hook(after)
        try:
            for row, ids in [pair for pair in inputs if pair[0]["split"] == "test"][:4]:
                for arm in [(0, 0)] + ([best] if best is not None else []):
                    launches.clear()
                    sample = generate(ids, arm)
                    detail = [dict(layers=d, inputs=n, gpu_event_ms=a.elapsed_time(b))
                              for d, n, a, b in launches]
                    assert not arm[0] or any(x["layers"] == arm[0] for x in detail)
                    profiles.append(dict(id=row["id"], arm=arm, seconds=sample["seconds"],
                                         tokens=sample["tokens"], forwards=detail))
        finally:
            pre_hook.remove()
            post_hook.remove()
    prep.pop("snapshot", None)
    args.out.write_text(json.dumps(dict(prepared=prep, torch=torch.__version__,
        transformers=transformers_version, device=dev, platform=platform.platform(),
        accelerator=torch.cuda.get_device_name() if cuda else None,
        accelerator_memory_bytes=torch.cuda.get_device_properties(0).total_memory if cuda else None,
        cuda_runtime=torch.version.cuda, seed=args.seed, max_new_tokens=args.tokens,
        context=args.context, repeats=args.repeats, selected=locked, profiles=profiles,
        whole_experiment_seconds=time.perf_counter() - started,
        arms=[dict(arm=m, calibration=cal[m], test=test[m]) for m in modes],
        scope="batch-1 greedy prefill + decode, HF 4.48.3 early exit, excludes tokenization and model load",
    ), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
