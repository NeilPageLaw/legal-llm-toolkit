"""
Evaluation module for legal LLMs.

Provides benchmarks and metrics specific to legal
language understanding and generation tasks.
"""

from legalkit.eval.benchmark import LegalBenchmark
from legalkit.eval.metrics import LegalMetrics

__all__ = [
    "LegalBenchmark",
    "LegalMetrics",
]
