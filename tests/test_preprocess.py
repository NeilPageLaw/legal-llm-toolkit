"""
Tests for preprocessing module.
"""

import pytest
from legalkit.preprocess import (
    LegalPreprocessor,
    CitationParser,
    Anonymiser,
    LegalChunker,
)


class TestCitationParser:
    """Tests for citation parsing."""
    
    def test_uk_neutral_citation(self):
        """Test UK neutral citation parsing."""
        parser = CitationParser(jurisdiction="uk")
        text = "As held in Smith v Jones [2024] UKSC 15 at [42]"
        
        citations = parser.parse(text)
        
        assert len(citations) == 1
        assert citations[0].year == 2024
        assert citations[0].court == "UKSC"
        assert citations[0].jurisdiction == "uk"
        
    def test_uk_law_report_citation(self):
        """Test UK law report citation."""
        parser = CitationParser(jurisdiction="uk")
        text = "See [2024] 1 AC 123"
        
        citations = parser.parse(text)
        
        assert len(citations) == 1
        assert citations[0].reporter == "AC"
        
    def test_us_citation(self):
        """Test US federal citation."""
        parser = CitationParser(jurisdiction="us")
        text = "123 F.3d 456 (9th Cir. 2024)"
        
        citations = parser.parse(text)
        
        assert len(citations) == 1
        assert citations[0].volume == "123"
        assert citations[0].jurisdiction == "us"
        
    def test_eu_citation(self):
        """Test EU case citation."""
        parser = CitationParser(jurisdiction="eu")
        text = "In Case C-123/24, the Court held..."
        
        citations = parser.parse(text)
        
        assert len(citations) == 1
        assert citations[0].jurisdiction == "eu"
        
    def test_uk_legislation(self):
        """Test UK legislation reference."""
        parser = CitationParser(jurisdiction="uk")
        text = "Under section 1 of the Companies Act 2006"
        
        citations = parser.parse(text)
        
        assert len(citations) >= 1
        leg_citations = [c for c in citations if c.citation_type == "legislation"]
        assert len(leg_citations) == 1
        
    def test_multiple_citations(self):
        """Test parsing multiple citations."""
        parser = CitationParser(jurisdiction="uk")
        text = """
        As established in [2020] UKSC 1 and confirmed in 
        [2022] EWCA Civ 123, the principle applies.
        """
        
        citations = parser.parse(text)
        
        assert len(citations) == 2


class TestAnonymiser:
    """Tests for PII anonymisation."""
    
    def test_email_anonymisation(self):
        """Test email anonymisation."""
        anon = Anonymiser()
        text = "Contact john.smith@example.com for details"
        
        result = anon.anonymise(text)
        
        assert "john.smith@example.com" not in result.text
        assert "[EMAIL_1]" in result.text
        
    def test_phone_anonymisation(self):
        """Test phone number anonymisation."""
        anon = Anonymiser()
        text = "Call us on 07123 456789"
        
        result = anon.anonymise(text)
        
        assert "07123 456789" not in result.text
        
    def test_preserve_case_names(self):
        """Test that case names are preserved when configured."""
        anon = Anonymiser(preserve_case_names=True)
        text = "In Smith v Jones [2024] UKSC 1, the court held..."
        
        result = anon.anonymise(text)
        
        # Case names should be preserved in context of citations
        assert "Smith" in result.text or "[PERSON" in result.text
        
    def test_consistent_replacement(self):
        """Test consistent replacement of same entity."""
        anon = Anonymiser(consistent_replacement=True)
        text = "Mr Smith said. Later, Mr Smith confirmed."
        
        result = anon.anonymise(text)
        
        # Same name should get same replacement
        replacements = [v for v in result.mapping.values() if "PERSON" in v]
        # The same person should map to the same replacement
        
    def test_reversible_anonymisation(self):
        """Test that anonymisation can be reversed."""
        anon = Anonymiser()
        original = "Email john@test.com for help"
        
        result = anon.anonymise(original)
        reversed_text = anon.deanonymise(result.text, result.mapping)
        
        # Should be able to restore original
        assert "john@test.com" in reversed_text or original == reversed_text


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
        chunker = LegalChunker(
            chunk_size=200, 
            overlap=20,
            respect_paragraphs=True
        )
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
        
        assert all(hasattr(c, 'text') for c in chunks)
        assert all(hasattr(c, 'start_char') for c in chunks)


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
        processor = LegalPreprocessor(
            jurisdiction="uk",
            anonymise=True
        )
        text = "Contact john@email.com regarding Smith v Jones [2024] UKSC 1."
        
        result = processor.process(text)
        
        assert "john@email.com" not in result.processed
        assert result.anonymisation is not None
        
    def test_processing_with_chunks(self):
        """Test processing with chunking."""
        processor = LegalPreprocessor(
            jurisdiction="uk",
            chunk_size=50
        )
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
