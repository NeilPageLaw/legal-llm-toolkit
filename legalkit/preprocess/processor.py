"""
Main legal text preprocessor combining all preprocessing capabilities.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from legalkit.preprocess.anonymiser import AnonymisationResult, Anonymiser
from legalkit.preprocess.chunker import LegalChunker
from legalkit.preprocess.citations import Citation, CitationParser, deduplicate

# Spaces, tabs and Unicode spaces (no-break, thin, ideographic ...).
_HORIZONTAL_SPACE = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
# Zero-width characters and byte order marks.
_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_PAGE_NUMBER_LINE = re.compile(
    r"^[ \t]*(?:[-–—][ \t]*)?\d{1,4}(?:[ \t]*[-–—])?[ \t]*$\n?", re.MULTILINE
)
_PAGE_X_OF_Y_LINE = re.compile(
    r"^[ \t]*Page[ \t]+\d+(?:[ \t]+of[ \t]+\d+)?[ \t]*$\n?", re.MULTILINE | re.IGNORECASE
)


@dataclass
class ProcessedDocument:
    """Result of processing a legal document."""

    original: str
    processed: str
    citations: list[Citation] = field(default_factory=list)
    anonymisation: AnonymisationResult | None = None
    chunks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialisation."""
        return {
            "original_length": len(self.original),
            "processed_length": len(self.processed),
            "citation_count": len(self.citations),
            "citations": [c.to_dict() for c in self.citations],
            "chunk_count": len(self.chunks),
            "anonymised": self.anonymisation is not None,
            "metadata": self.metadata,
        }


class LegalPreprocessor:
    """
    Main preprocessor for legal documents.

    Combines citation parsing, anonymisation, and chunking into
    a single pipeline for preparing legal text for LLM training.

    Example:
        >>> processor = LegalPreprocessor(jurisdiction="uk", anonymise=True)
        >>> result = processor.process(document_text)
        >>> print(f"Found {len(result.citations)} citations")
        >>> print(f"Created {len(result.chunks)} training chunks")
    """

    def __init__(
        self,
        jurisdiction: str = "uk",
        anonymise: bool = False,
        preserve_case_names: bool = True,
        preserve_dates: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int | None = None,
        normalise_citations: bool = True,
        normalise_whitespace: bool = True,
        remove_headers_footers: bool = False,
        anonymiser: Anonymiser | None = None,
    ):
        """
        Initialise the preprocessor.

        Args:
            jurisdiction: Target jurisdiction ('uk', 'us', 'eu')
            anonymise: Whether to anonymise PII
            preserve_case_names: Keep case names when anonymising
            preserve_dates: Keep dates when anonymising
            chunk_size: Token size for chunks
            chunk_overlap: Overlap between chunks in tokens (default: 10% of
                chunk_size, at most 50)
            normalise_citations: Standardise case citation format
            normalise_whitespace: Clean up whitespace
            remove_headers_footers: Attempt to remove page numbers and
                "Page X of Y" lines. Also removes lines holding only a
                number, which in some judgment layouts are paragraph numbers.
            anonymiser: A configured Anonymiser (e.g. with a salt or NER
                model). Implies anonymise=True.
        """
        self.jurisdiction = jurisdiction.lower()
        self.anonymise = anonymise or anonymiser is not None
        self.normalise_citations = normalise_citations
        self.normalise_whitespace = normalise_whitespace
        self.remove_headers_footers = remove_headers_footers

        # Initialise components
        self.citation_parser = CitationParser(jurisdiction=jurisdiction)
        if anonymiser is None and anonymise:
            anonymiser = Anonymiser(
                preserve_case_names=preserve_case_names, preserve_dates=preserve_dates
            )
        self.anonymiser = anonymiser
        self.chunker = LegalChunker(
            chunk_size=chunk_size, overlap=chunk_overlap, jurisdiction=jurisdiction
        )

    def process(
        self,
        text: str,
        extract_citations: bool = True,
        create_chunks: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ProcessedDocument:
        """
        Process a legal document through the full pipeline.

        Args:
            text: Raw legal document text
            extract_citations: Whether to parse citations
            create_chunks: Whether to create training chunks
            metadata: Optional metadata to attach

        Returns:
            ProcessedDocument with all processing results
        """
        processed = text
        citations = []
        anonymisation_result = None
        chunks = []

        # Step 1: Basic normalisation
        if self.normalise_whitespace:
            processed = self._normalise_whitespace(processed)

        if self.remove_headers_footers:
            processed = self._remove_headers_footers(processed)

        # Step 2: Extract citations
        if extract_citations:
            occurrences = self.citation_parser.parse(processed, unique=False)
            citations = deduplicate(occurrences)

            # Optionally normalise citations in text
            if self.normalise_citations:
                processed = self._normalise_citations_in_text(processed, occurrences)

        # Step 3: Anonymise PII
        if self.anonymiser:
            anonymisation_result = self.anonymiser.anonymise(processed)
            processed = anonymisation_result.text

        # Step 4: Create chunks for training
        if create_chunks:
            chunks = self.chunker.chunk(processed)

        return ProcessedDocument(
            original=text,
            processed=processed,
            citations=citations,
            anonymisation=anonymisation_result,
            chunks=chunks,
            metadata=metadata or {},
        )

    def process_batch(self, documents: list[str], **kwargs) -> list[ProcessedDocument]:
        """
        Process multiple documents.

        Args:
            documents: List of document texts
            **kwargs: Arguments passed to process()

        Returns:
            List of ProcessedDocument results
        """
        return [self.process(doc, **kwargs) for doc in documents]

    def _normalise_whitespace(self, text: str) -> str:
        """Normalise whitespace in document, keeping paragraph breaks."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Page breaks from PDF extraction become paragraph breaks.
        text = text.replace("\f", "\n\n").replace("\v", "\n")
        text = _INVISIBLE.sub("", text)
        text = _HORIZONTAL_SPACE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _remove_headers_footers(self, text: str) -> str:
        """Remove lines that hold only a page number or "Page X of Y"."""
        text = _PAGE_NUMBER_LINE.sub("", text)
        text = _PAGE_X_OF_Y_LINE.sub("", text)
        return re.sub(r"\n{3,}", "\n\n", text)

    def _normalise_citations_in_text(self, text: str, citations: list[Citation]) -> str:
        """
        Replace every case citation in text with its normalised form.

        Legislation references are left as written: their normalised form
        reorders the words ("Companies Act 2006, s 1"), which would break
        the grammar of the surrounding sentence.

        Args:
            text: The text the citations were parsed from.
            citations: Every occurrence, with offsets into ``text``.
        """
        result = text
        replaceable = [
            c
            for c in citations
            if c.citation_type == "case" and c.normalised and c.start is not None
        ]
        # Replace from the end so earlier offsets stay valid.
        for citation in sorted(replaceable, key=lambda c: c.start or 0, reverse=True):
            start, end = citation.start or 0, citation.end or 0
            if text[start:end] == citation.raw:
                result = result[:start] + citation.normalised + result[end:]

        return result

    def get_statistics(self, result: ProcessedDocument) -> dict[str, Any]:
        """
        Get statistics about processed document.

        Args:
            result: ProcessedDocument to analyse

        Returns:
            Dictionary of statistics
        """
        stats = {
            "original_chars": len(result.original),
            "processed_chars": len(result.processed),
            "reduction_pct": round((1 - len(result.processed) / len(result.original)) * 100, 2)
            if result.original
            else 0,
            "citation_count": len(result.citations),
            "citations_by_type": {},
            "citations_by_jurisdiction": {},
            "chunk_count": len(result.chunks),
        }

        # Count citations by type and jurisdiction
        for c in result.citations:
            stats["citations_by_type"][c.citation_type] = (
                stats["citations_by_type"].get(c.citation_type, 0) + 1
            )
            stats["citations_by_jurisdiction"][c.jurisdiction] = (
                stats["citations_by_jurisdiction"].get(c.jurisdiction, 0) + 1
            )

        # Anonymisation stats
        if result.anonymisation:
            stats["entities_anonymised"] = len(result.anonymisation.entities)
            stats["unique_entities"] = len(result.anonymisation.mapping)

        return stats
