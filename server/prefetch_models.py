#!/usr/bin/env python3
"""
prefetch_models.py — Download model weights into the local HuggingFace cache.

Run this once after installing vLLM. The script is idempotent: already-cached
files are skipped. Partial downloads resume automatically on re-run. The vLLM
server reads from this cache at runtime and never re-downloads.

Usage:
    python prefetch_models.py                # download both models
    python prefetch_models.py --only qwen    # Qwen2.5-3B only
    python prefetch_models.py --only mistral # Mistral-7B-AWQ only
    python prefetch_models.py --verify       # check cache without downloading

Environment:
    HF_HOME    Override the HuggingFace cache location (default: ~/.cache/huggingface)
    HF_TOKEN   Auth token for gated repos. Not needed for the default models.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    estimated_size_gb: float
    notes: str


# The two models the client will let users pick between.
# If you want to swap variants, edit repo_id here.
MODELS: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        estimated_size_gb=6.2,
        notes="FP16, ~3B params, no quantization, ungated.",
    ),
    "mistral": ModelSpec(
        key="mistral",
        repo_id="solidrust/Mistral-7B-Instruct-v0.3-AWQ",
        estimated_size_gb=4.5,
        notes=(
            "AWQ INT4 quantized 7B, ungated. "
            "Fallback if this repo is unavailable: "
            "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
        ),
    ),
}

# Files we never need for inference. Skipping these saves bandwidth and disk:
# many repos ship .bin (legacy PyTorch pickle) alongside .safetensors, plus
# framework-specific weights we'll never load.
SKIP_PATTERNS = [
    "*.bin",
    "*.msgpack",
    "*.h5",
    "*.onnx",
    "*.gguf",
    "*.pt",
    "*.pth",
    "training_args.bin",
    "optimizer.pt",
]


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def cache_path_for(repo_id: str) -> Path:
    """Local cache directory snapshot_download writes to for this repo."""
    return hf_home() / "hub" / f"models--{repo_id.replace('/', '--')}"


def cached_size_gb(repo_id: str) -> float:
    path = cache_path_for(repo_id)
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def free_space_gb() -> float:
    """Free space on the disk that holds HF_HOME."""
    target = hf_home()
    target.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(target).free / (1024 ** 3)


def download_model(spec: ModelSpec) -> bool:
    print(f"\n— {spec.repo_id}")
    print(f"  {spec.notes}")
    print(f"  estimated size: {spec.estimated_size_gb:.1f} GB")

    cached = cached_size_gb(spec.repo_id)
    if cached > 0:
        print(f"  already on disk: {cached:.2f} GB (will verify and complete if partial)")

    try:
        local_dir = snapshot_download(
            repo_id=spec.repo_id,
            ignore_patterns=SKIP_PATTERNS,
        )
    except RepositoryNotFoundError:
        print(
            f"  ERROR: repo not found on HuggingFace Hub.\n"
            f"  This usually means: (a) the repo was renamed or removed, or\n"
            f"  (b) it is gated and you need to run `huggingface-cli login` first."
        )
        return False
    except HfHubHTTPError as e:
        print(f"  ERROR: HTTP error from HuggingFace Hub: {e}")
        return False
    except KeyboardInterrupt:
        print("\n  Download interrupted. Re-run this script to resume.")
        sys.exit(130)

    final = cached_size_gb(spec.repo_id)
    print(f"  done — {final:.2f} GB cached at {local_dir}")
    return True


def verify_model(spec: ModelSpec) -> bool:
    """Heuristic check: present and at least half the expected size."""
    cached = cached_size_gb(spec.repo_id)
    threshold = spec.estimated_size_gb * 0.5
    if cached < threshold:
        print(f"  {spec.repo_id}: MISSING or partial ({cached:.2f} GB on disk)")
        return False
    print(f"  {spec.repo_id}: OK ({cached:.2f} GB on disk)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download model weights into the local HuggingFace cache.",
    )
    parser.add_argument(
        "--only",
        choices=list(MODELS.keys()),
        help="Download only one model (default: all).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check the cache without downloading.",
    )
    args = parser.parse_args()

    targets = [MODELS[args.only]] if args.only else list(MODELS.values())

    print(f"HF cache root: {hf_home()}")
    print(f"Free disk on cache volume: {free_space_gb():.1f} GB")

    if args.verify:
        print("\nVerifying local cache:")
        all_ok = all(verify_model(spec) for spec in targets)
        return 0 if all_ok else 1

    estimated_total = sum(s.estimated_size_gb for s in targets)
    print(f"Will fetch {len(targets)} model(s), ~{estimated_total:.1f} GB total.")
    print("This is a one-time download. Re-runs are no-ops if files are present.")

    failures: list[str] = []
    for spec in targets:
        if not download_model(spec):
            failures.append(spec.repo_id)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} model(s) did not download cleanly:")
        for repo in failures:
            print(f"  - {repo}")
        return 1
    print("All models cached. Next: run smoke_test.py to verify the engine loads them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())