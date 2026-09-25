"""
Evaluation module for legal LLMs.

Provides benchmarks and metrics specific to legal
language understanding and generation tasks.
"""

from legalkit.eval.benchmark import BenchmarkResult, BenchmarkSuite, LegalBenchmark
from legalkit.eval.metrics import LegalMetrics, rouge_l, token_f1

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "LegalBenchmark",
    "LegalMetrics",
    "rouge_l",
    "token_f1",
]
