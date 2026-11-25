# R119-P0 — Multi-layer sparse suffix feasibility screen

This is a real single-GPU feasibility run on the previously validated LayerSkip 1B checkpoint. The sparse paths execute a reduced set of deep-layer queries/MLPs on the RTX 3090; they do not replay dense teacher activations. Full findings and limitations are in [`record.md`](record.md).

## Reproduction

The run used Python 3.11, PyTorch 2.5.1+cu124, Transformers 4.48.3, one RTX 3090 24 GB, and the R118 local model snapshot. `prepared.json` and `prompts.json` pin the metadata and exact AG News document inputs; the model weights are referenced by SHA-256 and are not duplicated here.

```bash
python run.py \
  --root /path/to/hfl_sd_r118 \
  --out-dir results \
  --repeats 6 \
  --tokens 8 \
  --eval-docs 8

python stage_profile.py \
  --root /path/to/hfl_sd_r118 \
  --out results/stage_profile_final.jsonl \
  --repeats 5 \
  --tokens 8 \
  --docs 1
```

The R118 directory must contain `prepared.json`, `prompts.json`, and `layerskip-1b/model.safetensors`. The first six measured profile repetitions follow a two-token warmup; warmup hook events are cleared before measurement. The result files retain individual repetitions and per-document metrics. `stage_profile.py` separately records five requests per length for AR and LayerSkip stage attribution.

## Evidence boundary

The model was frozen: zero trainable parameters, zero optimizer steps, and zero training cost. The calibration and test documents are disjoint. This round did not run HFL, adapter tuning, sparse-model autoregressive decoding, or speculative decoding on the new sparse target. LayerSkip timing is the Transformers integrated self-speculative path. Per-layer and submodule CUDA-event timings are attribution aids, not removable wall-clock time. A separate five-repeat stage profile measures greedy argmax, Cache.update, candidate generation, target forward, and KV crop; see `results/stage_profile_final.jsonl`. These intervals overlap and are not additive. Multinomial sampling, isolated fallback recomputation time, new sparse-target AR/SD throughput, and reliable per-configuration peak memory remain unmeasured.
