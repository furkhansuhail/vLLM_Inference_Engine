#!/usr/bin/env python3
"""
smoke_test.py — Verify vLLM can load each model and generate text.

Standalone test of the engine + GPU + model stack. No FastAPI, no networking.
If this works, we know the inference path is healthy and any subsequent server
bug is in the web layer, not the engine.

Pins inference to GPU 1 (RTX 3080, 10 GB) via CUDA_VISIBLE_DEVICES so the
laptop's internal 4070 stays free for the desktop session.

Usage:
    python smoke_test.py              # test both models
    python smoke_test.py --only qwen
    python smoke_test.py --only mistral
"""

from __future__ import annotations

# CRITICAL: CUDA_VISIBLE_DEVICES must be set BEFORE importing torch or vllm.
# Once CUDA initializes, the visible-device set is locked in for the process.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # RTX 3080 only

import argparse
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TestSpec:
    key: str
    repo_id: str
    quantization: str | None   # None = FP16, "awq" = AWQ INT4
    max_model_len: int          # cap context size to keep KV cache reasonable


SPECS: dict[str, TestSpec] = {
    "qwen": TestSpec(
        key="qwen",
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        quantization=None,
        max_model_len=8192,
    ),
    "mistral": TestSpec(
        key="mistral",
        repo_id="solidrust/Mistral-7B-Instruct-v0.3-AWQ",
        quantization="awq",
        max_model_len=8192,
    ),
}

# Short prompt that exercises real generation without taking forever.
TEST_PROMPT = "Explain how a CPU cache hierarchy works in three sentences."

# Generation knobs. Deterministic-ish: low temperature for reproducible smoke test
# behavior across runs, but not greedy — greedy can produce degenerate loops in
# small models which would make the smoke test misleadingly look broken.
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 200


def build_chat_prompt(llm, user_text: str) -> str:
    """
    Apply the model's chat template via its tokenizer. This wraps the user
    text in the instruction format the model was trained on (ChatML for Qwen,
    [INST]...[/INST] for Mistral), giving meaningfully better outputs than
    raw text.
    """
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_test(spec: TestSpec) -> bool:
    """Load the model, generate once, report timing. Returns True on success."""
    # Heavy imports deferred so --help is fast and a failed --only on one
    # model doesn't pay vllm init cost for the other.
    from vllm import LLM, SamplingParams
    import torch

    print(f"\n{'=' * 64}")
    print(f"Testing: {spec.repo_id}")
    print(f"{'=' * 64}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available to PyTorch.")
        print("  Check: nvidia-smi works, driver installed, torch built with CUDA.")
        return False

    # After CUDA_VISIBLE_DEVICES=1, the 3080 is exposed as cuda:0 to this process.
    visible_count = torch.cuda.device_count()
    print(f"CUDA device(s) visible: {visible_count}")
    print(f"  cuda:0 = {torch.cuda.get_device_name(0)}")
    if visible_count != 1:
        print(f"  WARNING: expected 1 visible device (RTX 3080 only). "
              f"Got {visible_count}. Check CUDA_VISIBLE_DEVICES.")

    print(f"\nLoading model into VRAM...")
    t0 = time.perf_counter()

    try:
        llm = LLM(
            model=spec.repo_id,
            quantization=spec.quantization,
            dtype="auto",
            gpu_memory_utilization=0.85,   # ~8.5 GB of the 10 GB card
            max_model_len=spec.max_model_len,
            enforce_eager=False,           # use CUDA graphs (production path)
            trust_remote_code=False,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"ERROR: out of GPU memory loading {spec.repo_id}.")
        print(f"  Try: lower gpu_memory_utilization, or lower max_model_len.")
        print(f"  Detail: {e}")
        return False
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            print(f"ERROR: out of GPU memory (RuntimeError).")
            print(f"  Try: lower gpu_memory_utilization, or lower max_model_len.")
        elif "not found" in msg or "no such" in msg:
            print(f"ERROR: model files not found in HF cache.")
            print(f"  Run: python prefetch_models.py --only {spec.key}")
        else:
            print(f"ERROR (RuntimeError): {e}")
        return False
    except Exception as e:
        print(f"ERROR loading model: {type(e).__name__}: {e}")
        return False

    t_load = time.perf_counter() - t0
    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Loaded in {t_load:.1f}s.")
    print(f"VRAM: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved by PyTorch")

    # ---- Generate ----
    prompt = build_chat_prompt(llm, TEST_PROMPT)
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    print(f"\nPrompt: {TEST_PROMPT!r}")
    print("Generating...")

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    t_gen = time.perf_counter() - t0

    completion = outputs[0].outputs[0]
    n_tokens = len(completion.token_ids)
    tps = n_tokens / t_gen if t_gen > 0 else float("inf")
    finish = completion.finish_reason  # "stop" | "length" | etc.

    print(f"\nGenerated {n_tokens} tokens in {t_gen:.2f}s "
          f"({tps:.1f} tok/s, finish_reason={finish})")
    print(f"{'-' * 64}")
    print(completion.text.strip())
    print(f"{'-' * 64}")

    # Release VRAM before loading the next model. Without this, the second
    # LLM() call sees the first model's allocation still resident.
    del llm
    torch.cuda.empty_cache()
    mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"VRAM after cleanup: {mem_after:.2f} GB allocated")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify vLLM can load and generate with each model.",
    )
    parser.add_argument(
        "--only",
        choices=list(SPECS.keys()),
        help="Test only one model (default: both, in order).",
    )
    args = parser.parse_args()

    targets = [SPECS[args.only]] if args.only else list(SPECS.values())

    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print(f"Will smoke-test {len(targets)} model(s): "
          f"{[s.key for s in targets]}")

    failures: list[str] = []
    for spec in targets:
        if not run_test(spec):
            failures.append(spec.repo_id)

    print(f"\n{'=' * 64}")
    if failures:
        print(f"FAILED: {len(failures)} model(s) did not pass smoke test:")
        for repo in failures:
            print(f"  - {repo}")
        return 1
    print("All models passed. Engine + GPU + model stack verified.")
    print("Next: server.py — FastAPI wrapper with SSE token streaming.")
    return 0


if __name__ == "__main__":
    sys.exit(main())