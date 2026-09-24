"""
Tests for the preprocessing pipeline.
"""

import pytest

from legalkit.preprocess import Anonymiser, LegalPreprocessor


class TestLegalPreprocessor:
    def test_basic_processing(self):
        processor = LegalPreprocessor(jurisdiction="uk")
        result = processor.process("This is a legal document with Smith v Jones [2024] UKSC 1.")
        assert result.processed == "This is a legal document with Smith v Jones [2024] UKSC 1."
        assert len(result.citations) == 1
        assert result.citations[0].parties == "Smith v Jones"

    def test_case_citations_are_normalised_everywhere(self):
        text = "See [2023] ewca civ 5. Later: [2023]  ewca civ 5 at [3]."
        result = LegalPreprocessor().process(text)
        assert result.processed == "See [2023] EWCA Civ 5. Later: [2023] EWCA Civ 5 at [3]."
        assert len(result.citations) == 1

    def test_legislation_wording_is_kept(self):
        text = "Under section 1 of the Companies Act 2006, a company may act."
        result = LegalPreprocessor().process(text)
        assert result.processed == text
        assert result.citations[0].normalised == "Companies Act 2006, s 1"

    def test_normalisation_can_be_disabled(self):
        text = "See [2023] ewca civ 5."
        assert LegalPreprocessor(normalise_citations=False).process(text).processed == text

    def test_processing_with_anonymisation(self):
        processor = LegalPreprocessor(jurisdiction="uk", anonymise=True)
        result = processor.process("Contact john@email.com regarding Smith v Jones [2024] UKSC 1.")
        assert result.processed == "Contact [EMAIL_1] regarding Smith v Jones [2024] UKSC 1."
        assert result.anonymisation is not None

    def test_custom_anonymiser(self):
        processor = LegalPreprocessor(anonymiser=Anonymiser(entity_types=["email"]))
        assert processor.anonymise
        result = processor.process("Mr Smith emailed a@b.com")
        assert result.processed == "Mr Smith emailed [EMAIL_1]"

    def test_processing_with_chunks(self):
        processor = LegalPreprocessor(jurisdiction="uk", chunk_size=50)
        result = processor.process("This is a test document. " * 20, create_chunks=True)
        assert len(result.chunks) > 1

    def test_no_chunks_unless_requested(self):
        assert LegalPreprocessor().process("Text. " * 500).chunks == []

    def test_unknown_jurisdiction(self):
        with pytest.raises(ValueError):
            LegalPreprocessor(jurisdiction="xx")

    def test_statistics(self):
        processor = LegalPreprocessor(jurisdiction="uk")
        result = processor.process("See [2024] UKSC 1 and [2023] EWCA Civ 123.")
        stats = processor.get_statistics(result)
        assert stats["citation_count"] == 2
        assert stats["citations_by_type"] == {"case": 2}
        assert stats["citations_by_jurisdiction"] == {"uk": 2}

    def test_process_batch(self):
        results = LegalPreprocessor().process_batch(["[2020] UKSC 1", "No citations."])
        assert [len(r.citations) for r in results] == [1, 0]

    def test_to_dict(self):
        result = LegalPreprocessor().process("[2020] UKSC 1", metadata={"id": 7})
        record = result.to_dict()
        assert record["citation_count"] == 1
        assert record["metadata"] == {"id": 7}


class TestWhitespace:
    def test_normalise_whitespace(self):
        processor = LegalPreprocessor()
        text = "A  b\t\tc\r\n\r\n\r\n  \n\nd\fe​"
        assert processor.process(text).processed == "A b c\n\nd\n\ne"

    def test_remove_headers_footers_keeps_numbered_paragraphs(self):
        processor = LegalPreprocessor(remove_headers_footers=True)
        text = "Para one.\n\n- 3 -\n\n12 The court held.\nPage 2 of 9\nMore text."
        assert processor.process(text).processed == "Para one.\n\n12 The court held.\nMore text."
