#!/usr/bin/env python3
"""
hardware_sanity_check.py — VRAM breakdown for vLLM V1.

Reads weights from model_runner.model_memory_usage and sums the per-layer
KV cache tensors directly off model_runner.kv_caches. All measurements come
from inside the EngineCore worker process via collective_rpc, since that's
the only process where the model and torch allocator state live in V1.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"               # RTX 3080
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # pickle RPC fn

import logging
from pathlib import Path
from dotenv import load_dotenv
from vllm import LLM

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
if os.getenv("HF_TOKEN"):
    print("Found HF_TOKEN in .env")

logging.basicConfig(level=logging.INFO)

llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    max_model_len=4096,
    gpu_memory_utilization=0.80,
    enforce_eager=True,
)


def _probe(self):
    """Runs in the EngineCore worker. `self` is the Worker."""
    import torch

    mr = self.model_runner

    # Weights — exposed directly by the model runner.
    weights_bytes = int(getattr(mr, "model_memory_usage", 0))

    # KV cache — sum the actual tensors. kv_caches can be a flat list of
    # tensors, or a list of (key, value) pairs depending on the backend, so
    # we walk it defensively.
    def _tensor_bytes(t):
        return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0

    kv_bytes = 0
    kv_caches = getattr(mr, "kv_caches", None) or []
    for item in kv_caches:
        if isinstance(item, torch.Tensor):
            kv_bytes += _tensor_bytes(item)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                kv_bytes += _tensor_bytes(sub)

    # Whole-GPU view (driver-level) and torch allocator state.
    free_mem, total_mem = torch.cuda.mem_get_info()
    return {
        "weights":   weights_bytes,
        "kv_cache":  int(kv_bytes),
        "total_mem": int(total_mem),
        "free_mem":  int(free_mem),
        "reserved":  int(torch.cuda.memory_reserved()),
        "allocated": int(torch.cuda.memory_allocated()),
        "n_kv_layers": len(kv_caches),
    }


info = llm.collective_rpc(_probe)[0]

GIB = 1024 ** 3
total      = info["total_mem"] / GIB
free       = info["free_mem"]  / GIB
used       = total - free
reserved   = info["reserved"]  / GIB
allocated  = info["allocated"] / GIB
weights    = info["weights"]   / GIB
kv_cache   = info["kv_cache"]  / GIB
# Cross-check: anything torch is holding that isn't weights or KV cache.
other_torch = max(0.0, allocated - weights - kv_cache)
non_torch   = max(0.0, used - reserved)  # CUDA ctx, cuBLAS workspaces, NCCL...

print("\n" + "=" * 56)
print("VRAM BREAKDOWN — Qwen2.5-3B-Instruct on RTX 3080")
print("=" * 56)
print(f"GPU total:             {total:6.2f} GiB")
print(f"GPU free:              {free:6.2f} GiB")
print(f"GPU used (all procs):  {used:6.2f} GiB")
print("-" * 56)
print(f"  Model weights:       {weights:6.2f} GiB")
print(f"  KV cache ({info['n_kv_layers']:>2} layers): {kv_cache:6.2f} GiB")
print(f"  Other torch tensors: {other_torch:6.2f} GiB  (activations, scratch, etc.)")
print(f"  Torch reserved:      {reserved:6.2f} GiB  (allocator cached blocks)")
print(f"  Non-torch / driver:  {non_torch:6.2f} GiB  (CUDA ctx, cuBLAS, NCCL)")
print("=" * 56 + "\n")

# #!/usr/bin/env python3
# """
# hardware_sanity_check.py — Discover vLLM V1 memory attributes and report VRAM.
#
# Runs the memory probe INSIDE the EngineCore worker (via collective_rpc), which
# is the only process where torch.cuda.memory_allocated/reserved are meaningful
# under V1. Also enumerates every *memory* / *kv_cache* / *watermark* attribute
# on the worker and model_runner so we can pin the right names without guessing.
# """
#
# import os
#
# # CUDA env vars MUST be set before importing torch or vllm.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # RTX 3080
# os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # allow pickled RPC fn
#
# import logging
# from pathlib import Path
# from dotenv import load_dotenv
# from vllm import LLM
#
# # Load HF_TOKEN if present (not required for ungated models).
# env_path = Path(__file__).parent.parent / ".env"
# load_dotenv(dotenv_path=env_path)
# if os.getenv("HF_TOKEN"):
#     print("Found HF_TOKEN in .env")
#
# logging.basicConfig(level=logging.INFO)
#
# llm = LLM(
#     model="Qwen/Qwen2.5-3B-Instruct",
#     max_model_len=4096,
#     gpu_memory_utilization=0.80,
#     enforce_eager=True,
# )
#
#
# def _probe(self):
#     """
#     Executes inside the EngineCore worker process.
#     `self` is the Worker; self.model_runner is the GPUModelRunner.
#     """
#     import torch
#
#     mr = self.model_runner
#
#     # Walk attributes on both the worker and the model_runner. Keep anything
#     # whose name hints at memory accounting and whose value is a positive number.
#     keywords = ("memory", "kv_cache", "watermark", "bytes", "usage")
#     candidates: dict[str, int] = {}
#     for obj_name, obj in (("worker", self), ("model_runner", mr)):
#         for attr in dir(obj):
#             if attr.startswith("_"):
#                 continue
#             if not any(k in attr.lower() for k in keywords):
#                 continue
#             try:
#                 val = getattr(obj, attr)
#             except Exception:
#                 continue
#             if isinstance(val, (int, float)) and val > 0:
#                 candidates[f"{obj_name}.{attr}"] = int(val)
#
#     # Process-local CUDA accounting — meaningful here because the model lives
#     # in THIS process. mem_get_info reads the driver (whole-GPU view).
#     free_mem, total_mem = torch.cuda.mem_get_info()
#     return {
#         "candidates": candidates,
#         "total_mem": int(total_mem),
#         "free_mem":  int(free_mem),
#         "reserved":  int(torch.cuda.memory_reserved()),
#         "allocated": int(torch.cuda.memory_allocated()),
#     }
#
#
# info = llm.collective_rpc(_probe)[0]
#
# GIB = 1024 ** 3
# total_gib     = info["total_mem"] / GIB
# free_gib      = info["free_mem"]  / GIB
# reserved_gib  = info["reserved"]  / GIB
# allocated_gib = info["allocated"] / GIB
# used_gib      = total_gib - free_gib            # whole GPU, all processes
# non_torch_gib = max(0, used_gib - reserved_gib) # driver ctx, cuBLAS, NCCL...
#
# print("\n" + "=" * 56)
# print("DISCOVERED MEMORY ATTRIBUTES (positive numeric only)")
# print("=" * 56)
# if info["candidates"]:
#     width = max(len(k) for k in info["candidates"])
#     for name, val in sorted(info["candidates"].items()):
#         print(f"  {name:<{width}}  {val/GIB:7.3f} GiB  ({val:>13,} B)")
# else:
#     print("  (none found — attribute names changed or model didn't load)")
#
# print("\n" + "=" * 56)
# print("VRAM REPORT (read from inside the worker process)")
# print("=" * 56)
# print(f"GPU Total VRAM:        {total_gib:6.2f} GiB")
# print(f"GPU Free:              {free_gib:6.2f} GiB")
# print(f"GPU Used (all procs):  {used_gib:6.2f} GiB")
# print("-" * 56)
# print(f"  Torch allocated:     {allocated_gib:6.2f} GiB  (live tensors)")
# print(f"  Torch reserved:      {reserved_gib:6.2f} GiB  (cached blocks, ≥ allocated)")
# print(f"  Non-torch / driver:  {non_torch_gib:6.2f} GiB  (CUDA ctx, cuBLAS, NCCL, ...)")
# print("=" * 56 + "\n")