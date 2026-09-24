"""
Tests for preprocessing module.
"""

import pytest

from legalkit.preprocess import (
    LegalChunker,
    LegalPreprocessor,
)


class TestLegalChunker:
    """Tests for legal text chunking."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = LegalChunker(chunk_size=100, overlap=10)
        text = "This is a test. " * 50

        chunks = chunker.chunk(text)

        assert len(chunks) > 1

    def test_respects_paragraph_boundaries(self):
        """Test that chunking respects paragraphs."""
        chunker = LegalChunker(chunk_size=200, overlap=20, respect_paragraphs=True)
        text = """
        1.1 First paragraph of the agreement.
        
        1.2 Second paragraph with more details about the terms.
        
        1.3 Third paragraph explaining the conditions.
        """

        chunks = chunker.chunk(text)

        # Chunks should exist
        assert len(chunks) >= 1

    def test_chunk_with_metadata(self):
        """Test chunking with metadata."""
        chunker = LegalChunker(chunk_size=100)
        text = "SECTION 1\nContent here.\n\nSECTION 2\nMore content."

        chunks = chunker.chunk_with_metadata(text)

        assert all(hasattr(c, "text") for c in chunks)
        assert all(hasattr(c, "start_char") for c in chunks)


class TestLegalPreprocessor:
    """Tests for main preprocessor."""

    def test_basic_processing(self):
        """Test basic document processing."""
        processor = LegalPreprocessor(jurisdiction="uk")
        text = "This is a legal document with Smith v Jones [2024] UKSC 1."

        result = processor.process(text)

        assert result.processed is not None
        assert len(result.citations) == 1

    def test_processing_with_anonymisation(self):
        """Test processing with anonymisation enabled."""
        processor = LegalPreprocessor(jurisdiction="uk", anonymise=True)
        text = "Contact john@email.com regarding Smith v Jones [2024] UKSC 1."

        result = processor.process(text)

        assert "john@email.com" not in result.processed
        assert result.anonymisation is not None

    def test_processing_with_chunks(self):
        """Test processing with chunking."""
        processor = LegalPreprocessor(jurisdiction="uk", chunk_size=50)
        text = "This is a test document. " * 20

        result = processor.process(text, create_chunks=True)

        assert len(result.chunks) > 0

    def test_statistics(self):
        """Test statistics generation."""
        processor = LegalPreprocessor(jurisdiction="uk")
        text = "See [2024] UKSC 1 and [2023] EWCA Civ 123."

        result = processor.process(text)
        stats = processor.get_statistics(result)

        assert "citation_count" in stats
        assert stats["citation_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
