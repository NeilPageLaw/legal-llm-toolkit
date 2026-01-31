"""
Legal LLM Toolkit - A framework for fine-tuning and evaluating LLMs on legal text.

Modules:
    - data: Dataset loaders and legal corpus handlers
    - preprocess: Citation parsing, anonymisation, and text chunking
    - finetune: Training configurations and LoRA/QLoRA wrappers
    - eval: Legal benchmarks and evaluation metrics
    - jurisdictions: Jurisdiction-specific rules and patterns
"""

__version__ = "0.1.0"
__author__ = "Legal LLM Toolkit Contributors"

from legalkit.preprocess import LegalPreprocessor, Anonymiser, CitationParser
from legalkit.finetune import LegalTrainer, LegalTrainingConfig
from legalkit.eval import LegalBenchmark, LegalMetrics
from legalkit.data import LegalDataset, load_legal_corpus

__all__ = [
    # Core classes
    "LegalPreprocessor",
    "Anonymiser", 
    "CitationParser",
    "LegalTrainer",
    "LegalTrainingConfig",
    "LegalBenchmark",
    "LegalMetrics",
    "LegalDataset",
    # Functions
    "load_legal_corpus",
]
