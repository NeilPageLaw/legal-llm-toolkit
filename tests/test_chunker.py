"""
Tests for legal text chunking.
"""

import pytest

from legalkit.preprocess import LegalChunker

CONTRACT = (
    "1.1 The Supplier shall deliver the Goods on the Delivery Date stated in the Order.\n\n"
    "1.2 The Customer shall pay the Price within thirty days of the date of the invoice.\n\n"
    "(a) Late payment accrues interest at four per cent above base rate."
)


def assert_exact_slices(text, chunks):
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text


class TestChunking:
    def test_basic_chunking(self):
        chunker = LegalChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk("This is a test. " * 50)
        assert len(chunks) > 1
        max_tokens = chunker.chunk_size + chunker.min_chunk_size
        assert all(chunker.estimate_tokens(c) <= max_tokens for c in chunks)

    def test_short_text_is_one_chunk(self):
        assert LegalChunker().chunk("  A short clause.  ") == ["A short clause."]

    def test_empty_text(self):
        assert LegalChunker().chunk("   \n ") == []

    def test_clause_markers_are_kept(self):
        chunks = LegalChunker(chunk_size=25, overlap=0).chunk(CONTRACT)
        assert chunks[0].startswith("1.1 The Supplier")
        assert any(c.startswith("1.2 The Customer") for c in chunks)
        assert any(c.startswith("(a) Late payment") for c in chunks)

    def test_paragraphs_are_not_glued_together(self):
        joined = " ".join(LegalChunker(chunk_size=25, overlap=0).chunk(CONTRACT))
        assert "Order.1.2" not in joined
        assert "invoice.Late" not in joined

    def test_chunks_are_exact_slices_with_metadata(self):
        chunks = LegalChunker(chunk_size=25, overlap=2).chunk_with_metadata(CONTRACT)
        assert len(chunks) == 3
        assert_exact_slices(CONTRACT, chunks)
        assert [c.paragraph for c in chunks] == ["1.1", "1.2", "(a)"]

    def test_overlap_starts_on_a_word_boundary(self):
        chunks = LegalChunker(chunk_size=30, overlap=5).chunk_with_metadata(CONTRACT)
        assert len(chunks) == 3
        for previous, chunk in zip(chunks, chunks[1:], strict=False):
            assert chunk.start_char < previous.end_char  # overlapping
            assert CONTRACT[chunk.start_char - 1].isspace()

    def test_no_overlap(self):
        chunks = LegalChunker(chunk_size=25, overlap=0).chunk_with_metadata(CONTRACT)
        for previous, chunk in zip(chunks, chunks[1:], strict=False):
            assert chunk.start_char >= previous.end_char

    def test_indented_markers_are_detected(self):
        text = """
        1.1 First paragraph of the agreement.

        1.2 Second paragraph with more details about the terms.

        1.3 Third paragraph explaining the conditions.
        """
        chunks = LegalChunker(chunk_size=15, overlap=0).chunk_with_metadata(text)
        assert [c.paragraph for c in chunks] == ["1.1", "1.2", "1.3"]
        assert_exact_slices(text, chunks)

    def test_sections_carry_their_heading(self):
        text = "PART 1\nDefinitions apply.\n\nPART 2\n" + "The tenant shall pay. " * 30
        chunks = LegalChunker(chunk_size=40, overlap=0).chunk_with_metadata(text)
        assert chunks[0].section == "PART 1"
        assert chunks[-1].section == "PART 2"

    def test_small_sections_are_packed_together(self):
        text = "\n".join(f"{i}. Clause number {i} applies." for i in range(1, 11))
        chunks = LegalChunker(chunk_size=512).chunk(text)
        assert chunks == [text]

    def test_long_sentences_are_split_at_whitespace(self):
        text = "word " * 500
        chunks = LegalChunker(chunk_size=50, overlap=0).chunk(text)
        assert len(chunks) > 1
        assert all(set(chunk.split()) == {"word"} for chunk in chunks)

    def test_abbreviations_do_not_end_sentences(self):
        chunker = LegalChunker(chunk_size=100)
        spans = chunker._sentence_spans(
            text := "Mr. Smith relied on s. 3 and U.S. law. The judge disagreed.",
            0,
            len(text),
            1000,
        )
        assert [text[a:b].strip() for a, b in spans] == [
            "Mr. Smith relied on s. 3 and U.S. law.",
            "The judge disagreed.",
        ]

    def test_eu_articles_are_sections(self):
        text = "Article 1\nSubject matter.\n\nArticle 2\nScope."
        chunker = LegalChunker(chunk_size=8, overlap=0, min_chunk_size=0, jurisdiction="eu")
        assert [c.section for c in chunker.chunk_with_metadata(text)] == ["Article 1", "Article 2"]

    def test_small_trailing_chunk_is_merged(self):
        text = ("The lessee covenants to repair. " * 12) + "\n\nEnd."
        chunks = LegalChunker(chunk_size=100, overlap=0, min_chunk_size=10).chunk(text)
        assert chunks[-1].endswith("End.")
        assert chunks[-1] != "End."

    def test_estimate_tokens(self):
        assert LegalChunker().estimate_tokens("x" * 40) == 10


class TestValidation:
    def test_overlap_defaults_scale_with_chunk_size(self):
        assert LegalChunker(chunk_size=512).overlap == 50
        assert LegalChunker(chunk_size=50).overlap == 5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"chunk_size": 0},
            {"chunk_size": 50, "overlap": 50},
            {"chunk_size": 50, "overlap": -1},
            {"chunk_size": 50, "min_chunk_size": 50},
        ],
    )
    def test_invalid_sizes_raise(self, kwargs):
        with pytest.raises(ValueError):
            LegalChunker(**kwargs)

    def test_formerly_infinite_configuration_terminates(self):
        # chunk_size == overlap used to loop forever; small overlaps must still work.
        chunks = LegalChunker(chunk_size=10, overlap=9).chunk("This is a test sentence. " * 20)
        assert chunks
