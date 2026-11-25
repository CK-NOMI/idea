"""Prepare pinned LayerSkip 1B or SmolLM2 checkpoints and AG News prompts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from transformers import AutoConfig, AutoTokenizer
import torch
import transformers

MODEL = "facebook/layerskip-llama3.2-1B"
HF_REVISION = "81deb0f88734409cca506bcefcfb5c6bd2667565"
MODEL_URL = f"https://www.modelscope.cn/api/v1/models/{MODEL}/repo?Revision=master&FilePath="
WEIGHTS_SHA = "ba9a27bc1fadc74d7ca576aa618e464aa006b1c56b4a940c668c980a302708ee"
TOKENIZER_SHA = "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"
BLOBS = {
    "config.json": "bb65e33cb3db2727698e755950cb8c1f5a8b9775",
    "generation_config.json": "9d6875bc4b1ad172618be34ec767b21ed35fda84",
    "tokenizer.json": "1c1d8d5c9024994f1d3b00f9662b8dd89ca13cf2",
    "tokenizer_config.json": "93623bc19ca3b4757444f2c6ec6a3ffea46938a9",
    "special_tokens_map.json": "02ee80b6196926a5ad790a004d9efd6ab1ba6542",
    "LICENSE": "99692fc3969c32ee00e0b20484d2693947d3d7c2",
}
SMOL_BLOBS = {
    "generation_config.json": "0fce861c328ff24830f3037d91ce773254447bf7",
    "tokenizer.json": "f922b1797f0c88e71addc8393787831f2477a4bd",
    "tokenizer_config.json": "d45192775d58298087a1fedf5967fe5b63b091ab",
    "special_tokens_map.json": "f6652f246cb895ca1edfb16d10b57917b266e335",
}
SMOL = {
    "135M": ("93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
             "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
             "ccf3dac2f33905d152adbdf3886b5e38a4eb7579"),
    "360M": ("f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
             "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f",
             "2c111af0f7d9845b3b9910d3d18f7cdd94bf16c4"),
}
DATA_REVISION = "555590db4219b1243abb1918effd6a7425a2d75f"
DATA_URL = f"https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/{DATA_REVISION}/data/ag_news_csv/"
DATA_SHA = {
    "train": "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e",
    "test": "521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699",
}
PREFIX = {
    "train": (1024, "199e6cfd2478399e8c2bb6ab74f795c8c476c6536125c22727201aa11e3402d2"),
    "test": (256, "65860ae613aa57967567aee39c633730337e25b0cfea0372a82da5fea8f52407"),
}


def digest(path, method, prefix=b""):
    h = hashlib.new(method, prefix)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, check):
    if dest.is_file() and check(dest):
        print("Verified existing", dest.name, flush=True)
        return
    part = dest.with_suffix(dest.suffix + ".partial")
    if not (part.is_file() and check(part)):
        options = ["curl", "-fSLsS", "--connect-timeout", "10", "--speed-time", "30",
                   "--speed-limit", "8192", "--max-time", "900", "-o", str(part)]
        for _ in range(4):
            if dest.is_file() and check(dest):
                print("Verified externally staged", dest.name, flush=True)
                return
            resume = ["-C", "-"] if part.is_file() else []
            result = subprocess.run([*options, *resume, url], check=False)
            if result.returncode == 0 and part.is_file() and check(part):
                break
            if result.returncode in (0, 33):
                # A completed corrupt file or a server without range support.
                part.unlink(missing_ok=True)
        else:
            raise ValueError(f"Download or pinned checksum failed for {dest.name}")
    part.replace(dest)
    print("Ready", dest.name, dest.stat().st_size, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=("layerskip-1b", *SMOL), default="layerskip-1b")
    args = p.parse_args()
    root = args.root.resolve()
    snapshot = root / args.model
    snapshot.mkdir(parents=True, exist_ok=True)
    if args.model == "layerskip-1b":
        model_id, revision, weight_sha, blobs, base = MODEL, HF_REVISION, WEIGHTS_SHA, BLOBS, MODEL_URL
    else:
        model_id = f"HuggingFaceTB/SmolLM2-{args.model}"
        revision, weight_sha, config_blob = SMOL[args.model]
        blobs = dict(SMOL_BLOBS, **{"config.json": config_blob})
        base = f"https://huggingface.co/{model_id}/resolve/{revision}/"
    sources = {}

    def small(name):
        def matches(path):
            if args.model == "layerskip-1b" and name == "tokenizer.json":
                return digest(path, "sha256") == TOKENIZER_SHA
            prefix = f"blob {path.stat().st_size}\0".encode()
            return digest(path, "sha1", prefix) == blobs[name]
        download(base + name, snapshot / name, matches)

    with ThreadPoolExecutor(max_workers=6) as pool:
        weights = pool.submit(download, base + "model.safetensors",
                              snapshot / "model.safetensors",
                              lambda path: digest(path, "sha256") == weight_sha)
        list(pool.map(small, blobs))
        for split in ("train", "test"):
            prefix = root / (split + ".prefix.csv")
            if prefix.is_file() and digest(prefix, "sha256") == PREFIX[split][1]:
                sources[split] = prefix
                print("Verified pinned source prefix", split, PREFIX[split][0], flush=True)
            else:
                sources[split] = root / (split + ".csv")
                download(DATA_URL + split + ".csv", sources[split],
                         lambda path: digest(path, "sha256") == DATA_SHA[split])
        weights.result()

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    assert config.model_type == "llama"
    rows = []
    for split, role in [("train", "calibration"), ("test", "test")]:
        counts = [0] * 4
        with sources[split].open(newline="", encoding="utf-8") as source:
            for i, (label, title, description) in enumerate(csv.reader(source)):
                client = int(label) - 1
                text = title + "\n" + description
                if counts[client] == 8 or len(tokenizer(text).input_ids) < 24:
                    continue
                rows.append(dict(id=f"{split}-{i}", split=role, client=client,
                                 region=client // 2, text=text))
                counts[client] += 1
                if min(counts) == 8:
                    break
        assert min(counts) == 8
    prompts = root / "prompts.json"
    prompts.write_text(json.dumps(rows, ensure_ascii=False))
    prepared = dict(snapshot=str(snapshot), model_id=model_id, revision=revision,
                    weight_sha256=weight_sha, official_git_blobs=blobs,
                    tokenizer_lfs_sha256=TOKENIZER_SHA if args.model == "layerskip-1b" else None,
                    source_url=base, torch=torch.__version__,
                    transformers=transformers.__version__, n_layers=config.num_hidden_layers,
                    prompt_sha256=digest(prompts, "sha256"), dataset="AG News CSV",
                    dataset_revision=DATA_REVISION, rows=len(rows), ready=True,
                    dataset_sha256={split: digest(path, "sha256") for split, path in sources.items()},
                    dataset_source_rows={split: PREFIX[split][0] if path.name.endswith(".prefix.csv") else None
                                         for split, path in sources.items()})
    (root / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")


if __name__ == "__main__":
    main()
