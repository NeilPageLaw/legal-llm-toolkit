"""
Main legal text preprocessor combining all preprocessing capabilities.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from legalkit.preprocess.citations import CitationParser, Citation
from legalkit.preprocess.anonymiser import Anonymiser, AnonymisationResult
from legalkit.preprocess.chunker import LegalChunker


@dataclass
class ProcessedDocument:
    """Result of processing a legal document."""
    original: str
    processed: str
    citations: List[Citation] = field(default_factory=list)
    anonymisation: Optional[AnonymisationResult] = None
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation."""
        return {
            "original_length": len(self.original),
            "processed_length": len(self.processed),
            "citation_count": len(self.citations),
            "citations": [c.to_dict() for c in self.citations],
            "chunk_count": len(self.chunks),
            "anonymised": self.anonymisation is not None,
            "metadata": self.metadata
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
        chunk_overlap: int = 50,
        normalise_citations: bool = True,
        normalise_whitespace: bool = True,
        remove_headers_footers: bool = False
    ):
        """
        Initialise the preprocessor.
        
        Args:
            jurisdiction: Target jurisdiction ('uk', 'us', 'eu')
            anonymise: Whether to anonymise PII
            preserve_case_names: Keep case names when anonymising
            preserve_dates: Keep dates when anonymising
            chunk_size: Token size for chunks
            chunk_overlap: Overlap between chunks
            normalise_citations: Standardise citation format
            normalise_whitespace: Clean up whitespace
            remove_headers_footers: Attempt to remove page headers/footers
        """
        self.jurisdiction = jurisdiction.lower()
        self.anonymise = anonymise
        self.normalise_citations = normalise_citations
        self.normalise_whitespace = normalise_whitespace
        self.remove_headers_footers = remove_headers_footers
        
        # Initialise components
        self.citation_parser = CitationParser(jurisdiction=jurisdiction)
        self.anonymiser = Anonymiser(
            preserve_case_names=preserve_case_names,
            preserve_dates=preserve_dates
        ) if anonymise else None
        self.chunker = LegalChunker(
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            jurisdiction=jurisdiction
        )
        
    def process(
        self,
        text: str,
        extract_citations: bool = True,
        create_chunks: bool = False,
        metadata: Optional[Dict[str, Any]] = None
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
            citations = self.citation_parser.parse(processed)
            
            # Optionally normalise citations in text
            if self.normalise_citations:
                processed = self._normalise_citations_in_text(processed, citations)
        
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
            metadata=metadata or {}
        )
    
    def process_batch(
        self,
        documents: List[str],
        **kwargs
    ) -> List[ProcessedDocument]:
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
        """Normalise whitespace in document."""
        import re
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalise line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _remove_headers_footers(self, text: str) -> str:
        """Attempt to remove page headers and footers."""
        import re
        
        # Remove common page number patterns
        text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
        text = re.sub(r'\nPage \d+ of \d+\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove common header patterns (case numbers, dates at start of pages)
        # This is heuristic and may need tuning
        
        return text
    
    def _normalise_citations_in_text(
        self, 
        text: str, 
        citations: List[Citation]
    ) -> str:
        """Replace citations with normalised versions."""
        # Sort by position (reverse) to replace from end
        sorted_citations = sorted(
            [(c, text.find(c.raw)) for c in citations if c.raw in text],
            key=lambda x: x[1],
            reverse=True
        )
        
        result = text
        for citation, pos in sorted_citations:
            if pos >= 0 and citation.normalised:
                result = result[:pos] + citation.normalised + result[pos + len(citation.raw):]
                
        return result
    
    def get_statistics(self, result: ProcessedDocument) -> Dict[str, Any]:
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
            "reduction_pct": round(
                (1 - len(result.processed) / len(result.original)) * 100, 2
            ) if result.original else 0,
            "citation_count": len(result.citations),
            "citations_by_type": {},
            "citations_by_jurisdiction": {},
            "chunk_count": len(result.chunks),
        }
        
        # Count citations by type and jurisdiction
        for c in result.citations:
            stats["citations_by_type"][c.citation_type] = \
                stats["citations_by_type"].get(c.citation_type, 0) + 1
            stats["citations_by_jurisdiction"][c.jurisdiction] = \
                stats["citations_by_jurisdiction"].get(c.jurisdiction, 0) + 1
        
        # Anonymisation stats
        if result.anonymisation:
            stats["entities_anonymised"] = len(result.anonymisation.entities)
            stats["unique_entities"] = len(result.anonymisation.mapping)
            
        return stats
