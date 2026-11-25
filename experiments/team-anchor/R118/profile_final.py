"""Profile the final-layer share of dense cached autoregressive GPU inference."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    root = args.root
    prepared = json.loads((root / "prepared.json").read_text())
    config = prepared["snapshot"]
    tok = AutoTokenizer.from_pretrained(config, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True,
    ).to("cuda").eval()
    rows = [r for r in json.loads((root / "prompts.json").read_text()) if r["split"] == "test"]
    inputs = [
        (r["id"], tok(r["text"], return_tensors="pt", truncation=True, max_length=64).to("cuda"))
        for c in range(4) for r in [x for x in rows if x["client"] == c][:2]
    ]

    def event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    phases = []

    def before(_, positional):
        phases.append((positional[0].shape[1], event(), None))

    def after(*_):
        n, start, _ = phases[-1]
        phases[-1] = (n, start, event())

    def generate(ids):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        return time.perf_counter() - start, out[0, ids.input_ids.shape[1]:].tolist()

    generate(inputs[0][1])
    observations = []
    final = model.model.layers[-1]
    for doc, ids in inputs:
        plain, expected = generate(ids)
        phases.clear()
        hooks = [final.register_forward_pre_hook(before), final.register_forward_hook(after)]
        try:
            timed, actual = generate(ids)
        finally:
            for hook in hooks:
                hook.remove()
        assert expected == actual
        observations.append(dict(id=doc, tokens=len(actual), uninstrumented_seconds=plain,
            instrumented_seconds=timed, layer_invocations=[
                dict(input_tokens=n, cuda_event_ms=start.elapsed_time(end))
                for n, start, end in phases
            ]))
    layer_seconds = sum(x["cuda_event_ms"] for r in observations for x in r["layer_invocations"]) / 1000
    instrumented = sum(x["instrumented_seconds"] for x in observations)
    fraction = layer_seconds / instrumented
    args.out.write_text(json.dumps(dict(model=prepared["model_id"], revision=prepared["revision"],
        accelerator=torch.cuda.get_device_name(), torch=torch.__version__,
        transformers=transformers_version, sequence_lengths=[64, 32], observations=observations,
        last_layer_event_seconds=layer_seconds, instrumented_total_seconds=instrumented,
        uninstrumented_total_seconds=sum(r["uninstrumented_seconds"] for r in observations),
        observed_event_fraction=fraction,
        idealized_speedup_eliminating_entire_final_layer=(1 / (1 - fraction) if fraction < 1 else None),
        scope="instrumented CUDA stream events and serial projection, not actual sparse speed"),
        indent=2) + "\n")
    print("Final-layer profiling complete", fraction, flush=True)


if __name__ == "__main__":
    main()
