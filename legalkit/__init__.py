"""
Legal LLM Toolkit - A framework for fine-tuning and evaluating LLMs on legal text.

Modules:
    - data: Dataset loaders and legal corpus handlers
    - preprocess: Citation parsing, anonymisation, and text chunking
    - finetune: Training configurations and LoRA/QLoRA wrappers
    - eval: Legal benchmarks and evaluation metrics
    - jurisdictions: Jurisdiction-specific rules and patterns

Importing the package needs no machine-learning libraries; training and
model evaluation load them on first use.
"""

__version__ = "0.1.0"
__author__ = "Legal LLM Toolkit Contributors"

from legalkit.data import LegalDataset, LegalSample, load_legal_corpus, to_instruction_format
from legalkit.eval import LegalBenchmark, LegalMetrics
from legalkit.finetune import LegalTrainer, LegalTrainingConfig
from legalkit.preprocess import Anonymiser, CitationParser, LegalChunker, LegalPreprocessor

__all__ = [
    # Core classes
    "LegalPreprocessor",
    "Anonymiser",
    "CitationParser",
    "LegalChunker",
    "LegalTrainer",
    "LegalTrainingConfig",
    "LegalBenchmark",
    "LegalMetrics",
    "LegalDataset",
    "LegalSample",
    # Functions
    "load_legal_corpus",
    "to_instruction_format",
]
