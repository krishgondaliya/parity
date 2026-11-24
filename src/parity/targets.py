"""The releases pinned for Phase 0. Must match the entry in docs/DECISIONS.md.

A test checks that every revision here also appears in docs/DECISIONS.md.
"""

from parity.release import Release

CALIBRATION = Release("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")

PHASE0_TARGETS: tuple[Release, ...] = (
    Release("thinkingmachines/Inkling-2-Mini", "7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6"),
    Release("tencent/Hy4-flash", "eccd3123eec73f2040f4c9ed0de1883593470510"),
    Release("deepseek-ai/DeepSeek-V4.1-Flash-1017", "08c8657f02291e11b6f6b63e0e16014f8e748518"),
)
