"""
Legal dataset handling.

Loads documents and instruction data from local files or the Hugging Face
Hub, and prepares them for fine-tuning.
"""

from legalkit.data.dataset import LegalDataset, LegalSample
from legalkit.data.formatting import to_instruction_format
from legalkit.data.loaders import load_legal_corpus

__all__ = [
    "LegalDataset",
    "LegalSample",
    "load_legal_corpus",
    "to_instruction_format",
]
