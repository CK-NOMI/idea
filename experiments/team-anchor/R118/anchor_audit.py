"""Check exact one-layer anchor states and measure the error from dropping earlier layer states."""
import argparse
import json
from pathlib import Path
import platform
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import eager_attention_forward, rotate_half


def sparse_anchors(model, shallow, anchors, positions):
    layer = model.model.layers[-1]
    attn = layer.self_attn
    normed = layer.input_layernorm(shallow)

    def project(projection, x):
        return projection(x).reshape(1, x.shape[1], -1, attn.head_dim).transpose(1, 2)

    q = project(attn.q_proj, normed[:, anchors])
    k = project(attn.k_proj, normed)
    v = project(attn.v_proj, normed)
    cos, sin = model.model.rotary_emb(normed, positions[None])
    # Query RoPE is applied only at anchors. Key RoPE is applied at every position.
    q = q * cos[:, None, anchors] + rotate_half(q) * sin[:, None, anchors]
    k = k * cos[:, None] + rotate_half(k) * sin[:, None]
    visible = positions[None, :] <= anchors[:, None]
    mask = torch.zeros(visible.shape, device=shallow.device, dtype=shallow.dtype)
    mask = mask.masked_fill(~visible, torch.finfo(shallow.dtype).min)[None, None]
    y, _ = eager_attention_forward(attn, q, k, v, mask, attn.scaling)
    h = shallow[:, anchors] + attn.o_proj(y.reshape(1, len(anchors), -1))
    return h + layer.mlp(layer.post_attention_layernorm(h))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--docs", type=int, default=12)
    args = p.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    prep = json.loads((args.root / "prepared.json").read_text())
    rows = [r for r in json.loads((args.root / "prompts.json").read_text()) if r["split"] == "test"]
    rows = [r for c in range(4) for r in [a for a in rows if a["client"] == c][:args.docs // 4]]
    tok = AutoTokenizer.from_pretrained(prep["snapshot"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        prep["snapshot"], torch_dtype=torch.float32, use_safetensors=True,
        local_files_only=True, attn_implementation="eager",
    ).to(args.device).eval()
    record = []
    for row in rows:
        ids = tok(row["text"], return_tensors="pt", truncation=True,
                  max_length=args.context).input_ids.to(args.device)
        positions = torch.arange(ids.shape[-1], device=ids.device)
        captured = []
        handle = model.model.norm.register_forward_pre_hook(lambda _, xs: captured.append(xs[0]))
        try:
            features = model.model(ids, use_cache=False, output_hidden_states=True).hidden_states
        finally:
            handle.remove()
        x = features[-2]
        pre = features[-3]
        d = captured[0]
        for B in (2, 4, 8):
            a = positions[::B]
            exact = sparse_anchors(model, x, a, positions)
            skipped = pre.clone()
            skipped[:, a] = x[:, a]
            contaminated = sparse_anchors(model, skipped, a, positions)
            target = d[:, a]
            def predict(h):
                return model.lm_head(model.model.norm(h)).argmax(-1)
            best = predict(target)
            record.append(dict(id=row["id"], block=B, anchors=len(a),
                               exact_rms=(exact - target).square().mean().sqrt().item(),
                               exact_max=(exact - target).abs().max().item(),
                               exact_greedy_matches=(best == predict(exact)).sum().item(),
                               skipped_rms=(contaminated - target).square().mean().sqrt().item(),
                               skipped_max=(contaminated - target).abs().max().item(),
                               skipped_greedy_matches=(best == predict(contaminated)).sum().item()))
    args.out.write_text(json.dumps(dict(model=prep["model_id"], revision=prep["revision"],
        torch=torch.__version__, device=args.device, platform=platform.platform(),
        depth=model.config.num_hidden_layers, seconds=time.perf_counter() - started,
        scope="sparse final-layer anchor query prototype; float32; no end-to-end timings",
        rows=record), indent=2) + "\n")


if __name__ == "__main__":
    main()
