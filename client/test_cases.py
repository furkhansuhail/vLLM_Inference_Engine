"""
test_cases.py — The three demonstration scenarios.

Each case has three prompts that get fired concurrently against the server.
The cases are designed to exercise different aspects of vLLM's scheduler:

    A. Simple / Overlap   — short prompts with high word overlap.
                            Shows continuous batching efficiency and
                            (potentially) automatic prefix caching.

    B. Medium / Mixed     — two short prompts plus one medium outlier.
                            Shows how the batch handles uneven workloads.

    C. Complex / Divergent — three divergent long-form prompts.
                             Shows PagedAttention managing three independent
                             long sequences in the same KV cache pool.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestCase:
    key: str
    name: str
    description: str
    prompts: tuple[str, str, str]
    max_tokens: int


CASE_A = TestCase(
    key="A",
    name="Case A — Simple / Overlap",
    description=(
        "Three short prompts on identical topics with high word overlap. "
        "Should demonstrate efficient continuous batching, and (if vLLM's "
        "automatic prefix caching is active) reduced prefill time for the "
        "second and third requests."
    ),
    prompts=(
        "Explain quantum entanglement in two sentences.",
        "Explain quantum tunneling in two sentences.",
        "Explain quantum superposition in two sentences.",
    ),
    max_tokens=120,
)

CASE_B = TestCase(
    key="B",
    name="Case B — Medium / Mixed",
    description=(
        "Two short prompts on related networking topics plus one medium-length "
        "outlier requesting a detailed multi-paragraph answer. Demonstrates "
        "how continuous batching keeps the short requests moving while the "
        "long one continues — and how throughput drops once the short "
        "requests finish and the batch thins out."
    ),
    prompts=(
        "Describe how DNS resolution works in three sentences.",
        "Describe how a load balancer distributes traffic in three sentences.",
        (
            "Write a detailed technical explanation of how HTTPS works, "
            "covering the TLS 1.3 handshake, certificate validation, "
            "session key derivation, and the role of certificate authorities. "
            "Use at least four paragraphs."
        ),
    ),
    max_tokens=400,
)

CASE_C = TestCase(
    key="C",
    name="Case C — Complex / Divergent",
    description=(
        "Three completely divergent, long-form prompts running concurrently. "
        "This is the PagedAttention showcase — three independent long "
        "sequences sharing the same physical block pool with no fragmentation. "
        "Watch the GPU cache usage climb in the server metrics panel."
    ),
    prompts=(
        (
            "Write a 400-word analysis of the philosophical implications of "
            "free will versus determinism, drawing on perspectives from "
            "modern neuroscience, compatibilism, and classical philosophy."
        ),
        (
            "Compose a 400-word technical guide to building a microservices "
            "architecture for a financial trading platform. Cover resilience "
            "patterns, observability, regulatory considerations, and the "
            "tradeoffs versus a monolithic approach."
        ),
        (
            "Write a 400-word fictional short story about a marine biologist "
            "who discovers an intelligent species of cephalopod off the coast "
            "of Norway. Include character development, plot tension, and a "
            "surprise ending."
        ),
    ),
    max_tokens=600,
)


ALL_CASES: dict[str, TestCase] = {c.key: c for c in (CASE_A, CASE_B, CASE_C)}
