"""Measure residual transfer on frozen pretrained models and held-out public text."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = ("dense", "shallow", "bridge", "periodic", "raw", "innovation", "regional")


def sufficient(docs):
    x = torch.cat([row["x"] for row in docs]).float()
    r = torch.cat([row["d"] - row["x"] for row in docs]).float()
    return len(x), x.sum(0), r.sum(0), x.square().sum(0), (x * r).sum(0)


def fit(stats):
    n, x, r, x2, xr = map(sum, zip(*stats))
    var = (x2 - x.square() / n).clamp_min(0) / n
    slope = (xr / n - (x / n) * (r / n)) / (var + 0.1 * var.mean().clamp_min(1e-8))
    bias = (r - slope * x) / n
    return slope, bias


def predict(x, coefficients):
    slope, bias = coefficients
    return x * slope + bias


def state(x, d, a, correction, carry, gamma):
    t = torch.arange(len(x), device=x.device)
    h = x + correction + gamma * carry
    h[t == a] = d[t == a]
    return h


def score(logits, labels, ref):
    logp = logits.float().log_softmax(-1)
    nll = -logp.gather(-1, labels[:, None]).sum().item()
    kl = (ref.exp() * (ref - logp)).sum().item()
    overlap = torch.minimum(ref.exp(), logp.exp()).sum().item()
    return [nll, kl, overlap]


def paired_ci(rows, arm, baseline, seed):
    by = {}
    for row in rows:
        by.setdefault(row["id"], {})[row["arm"]] = row
    docs = [
        (items[arm]["nll"] - items[baseline]["nll"], items[arm]["positions"])
        for items in by.values()
    ]
    rng = random.Random(seed)
    values = []
    for _ in range(3000):
        sampled = rng.choices(docs, k=len(docs))
        values.append(sum(x for x, _ in sampled) / sum(n for _, n in sampled))
    values.sort()
    return dict(delta=sum(x for x, _ in docs) / sum(n for _, n in docs),
                interval=[values[75], values[2924]],
                wins=sum(x < 0 for x, _ in docs), docs=len(docs))


@torch.inference_mode()
def run(args):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    root = args.root
    prep = json.loads((root / "prepared.json").read_text())
    raw = (root / "prompts.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prep["prompt_sha256"]
    rows = json.loads(raw)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA unavailable"
    model = AutoModelForCausalLM.from_pretrained(
        prep["snapshot"], torch_dtype=dtype, use_safetensors=True,
        local_files_only=True,
    ).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(prep["snapshot"], local_files_only=True)
    depth = model.config.num_hidden_layers
    exits = args.exits or sorted({3, depth // 2, depth - 1})
    assert all(0 < e < depth for e in exits)
    docs = []
    counters = {}
    for row in rows:
        key = (row["split"], row["client"])
        i = counters.get(key, 0)
        counters[key] = i + 1
        if i >= args.per_class:
            continue
        ids = tok(row["text"], return_tensors="pt", truncation=True,
                  max_length=args.context).input_ids.to(args.device)
        captured = []
        handle = model.model.norm.register_forward_pre_hook(
            lambda _, inputs: captured.append(inputs[0].detach())
        )
        try:
            output = model.model(ids, use_cache=False, output_hidden_states=True)
        finally:
            handle.remove()
        assert len(captured) == 1
        role = row["split"]
        if role == "calibration":
            role = "fit" if i < args.per_class - 2 else "tune"
        docs.append(dict(id=row["id"], role=role, labels=ids[0, 1:],
                         client=row["client"], region=row["region"],
                         shallow={e: output.hidden_states[e][0, :-1].float() for e in exits},
                         d=captured[0][0, :-1].float()))
    print("Dense features extracted", len(docs), "documents", flush=True)

    def logits(hidden):
        return model.lm_head(model.model.norm(hidden.to(dtype)))

    combined = []
    for exit_layer in exits:
        for row in docs:
            row["x"] = row["shallow"][exit_layer]
        stats = [sufficient([r for r in docs if r["role"] == "fit" and r["client"] == c])
                 for c in range(4)]
        pooled = fit(stats)
        regions = []
        for region in range(2):
            local = fit(stats[2 * region:2 * region + 2])
            regions.append(tuple((x + y) / 2 for x, y in zip(pooled, local)))
        for B in args.blocks:
            calibration = {}
            test_rows = []
            for row in docs:
                if row["role"] != "tune":
                    continue
                x, d, y = row["x"], row["d"], row["labels"]
                t = torch.arange(len(x), device=x.device)
                a = t // B * B
                lag = t - a
                residual = (d - x)[a]
                for arm, coefficients in [("raw", None), ("innovation", pooled),
                                          ("regional", regions[row["region"]])]:
                    base = torch.zeros_like(x) if coefficients is None else predict(x, coefficients)
                    carry = residual - base[a]
                    # All arms select the same number of coefficients on calibration labels only.
                    for g in GRID:
                        h = state(x, d, a, base, carry, g)
                        losses = F.cross_entropy(logits(h).float(), y, reduction="none")
                        for j in range(1, B):
                            region = row["region"] if arm == "regional" else -1
                            key = (arm, region, j, g)
                            calibration[key] = calibration.get(key, 0.0) + losses[lag == j].sum().item()

            def selected(arm, region):
                return [0.0] + [
                    min(GRID, key=lambda g: calibration[arm, region, j, g])
                    for j in range(1, B)
                ]

            coefficients = {
                "raw": selected("raw", -1),
                "innovation": selected("innovation", -1),
                "regional": [selected("regional", r) for r in range(2)],
            }
            for row in docs:
                if row["role"] != "test":
                    continue
                x, d, y = row["x"], row["d"], row["labels"]
                t = torch.arange(len(x), device=x.device)
                a = t // B * B
                lag = t - a
                residual = (d - x)[a]
                global_base = predict(x, pooled)
                regional_base = predict(x, regions[row["region"]])
                def gamma(values):
                    return torch.tensor(values, device=x.device)[lag, None]
                choices = {
                    "dense": d, "shallow": x, "bridge": x + global_base,
                    "periodic": state(x, d, a, 0, residual, 0),
                    "raw": state(x, d, a, 0, residual, gamma(coefficients["raw"])),
                    "innovation": state(x, d, a, global_base, residual - global_base[a],
                                        gamma(coefficients["innovation"])),
                    "regional": state(x, d, a, regional_base, residual - regional_base[a],
                                      gamma(coefficients["regional"][row["region"]])),
                }
                dense = logits(d).float().log_softmax(-1)
                for arm, h in choices.items():
                    nll, kl, overlap = score(logits(h), y, dense)
                    test_rows.append(dict(id=row["id"], client=row["client"], arm=arm,
                                          positions=len(y), anchors=(t == a).sum().item(),
                                          nll=nll, kl=kl, overlap=overlap))
            summary = {}
            for arm in ARMS:
                sample = [r for r in test_rows if r["arm"] == arm]
                n = sum(r["positions"] for r in sample)
                summary[arm] = dict(positions=n, anchors=sum(r["anchors"] for r in sample),
                                    **{m: sum(r[m] for r in sample) / n
                                       for m in ("nll", "kl", "overlap")})
            paired = {f"{a}_vs_{b}": paired_ci(test_rows, a, b, args.seed)
                      for a, b in [("raw", "dense"), ("innovation", "dense"),
                                   ("innovation", "raw"), ("innovation", "bridge"),
                                   ("innovation", "periodic"), ("regional", "innovation")]}
            print("exit", exit_layer, "block", B,
                  {arm: round(s["nll"], 5) for arm, s in summary.items()}, flush=True)
            combined.append(dict(exit=exit_layer, block=B, gamma=coefficients, summary=summary,
                                 paired_document_bootstrap_95=paired, test=test_rows))
    prep.pop("snapshot", None)
    return dict(prepared=prep, torch=torch.__version__, platform=platform.platform(),
                device=args.device, cuda_device=torch.cuda.get_device_name() if args.device == "cuda" else None,
                dtype=str(dtype), seed=args.seed, context=args.context, threads=args.threads,
                per_class=args.per_class, n_layers=depth, args={"exits": exits, "blocks": args.blocks},
                splits={role: [r["id"] for r in docs if r["role"] == role]
                        for role in ("fit", "tune", "test")},
                elapsed_for_experiment_seconds=time.perf_counter() - started,
                scope="dense teacher-forced traces, offline calibration, no sparse latency or real SD",
                results=combined)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=118)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--per-class", type=int, choices=range(4, 9), default=8)
    p.add_argument("--exits", type=int, nargs="+")
    p.add_argument("--blocks", type=int, nargs="+", default=[2, 4, 8])
    args = p.parse_args()
    result = run(args)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
