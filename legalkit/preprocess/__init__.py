"""
Legal text preprocessing module.

Handles citation parsing, anonymisation, chunking, and normalisation
of legal documents across multiple jurisdictions.
"""

from legalkit.preprocess.anonymiser import Anonymiser
from legalkit.preprocess.chunker import LegalChunker
from legalkit.preprocess.citations import CitationParser
from legalkit.preprocess.processor import LegalPreprocessor

__all__ = [
    "LegalPreprocessor",
    "Anonymiser",
    "CitationParser",
    "LegalChunker",
]
