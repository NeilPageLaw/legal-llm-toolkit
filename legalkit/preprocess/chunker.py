"""
Intelligent chunking for legal documents.

Preserves legal structure (sections, paragraphs, clauses) while
creating appropriately sized chunks for LLM training.
"""

import re
from dataclasses import dataclass

# Rough characters-per-token ratio for English text.
CHARS_PER_TOKEN = 4

# A full stop followed by whitespace and the start of a new sentence.
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[A-Z0-9(\[\"“‘])")

# Tokens ending in a full stop that do not end a sentence.
_ABBREVIATIONS = frozenset(
    """
    v vs mr mrs ms dr prof no nos s ss para paras art arts reg regs sch ch pt cl
    e.g i.e etc ltd co inc corp plc st cf viz al op cit ibid id ed eds vol p pp n nn
    fn j lj cj jj rt hon esq sec secs subs
    """.split()
)


@dataclass
class Chunk:
    """A chunk of legal text with metadata."""

    text: str
    start_char: int
    end_char: int
    section: str | None = None
    paragraph: str | None = None

    @property
    def length(self) -> int:
        return len(self.text)


class LegalChunker:
    """
    Intelligent chunker for legal documents.

    Unlike generic text chunkers, this preserves legal structure:
    - Keeps clauses together when possible
    - Respects section boundaries
    - Handles numbered paragraphs correctly
    - Maintains context across chunk boundaries

    Documents are split into sections, then paragraphs, then sentences,
    only as far as needed, and the pieces are packed into chunks of up to
    ``chunk_size`` tokens, overlap included. A remainder smaller than
    ``min_chunk_size`` is merged into its neighbour, which can take that
    chunk up to ``chunk_size + min_chunk_size`` tokens. Every chunk is an
    exact slice of the input: ``text[chunk.start_char:chunk.end_char] == chunk.text``.

    Example:
        >>> chunker = LegalChunker(chunk_size=512, overlap=50)
        >>> chunks = chunker.chunk_with_metadata(contract_text)
        >>> for chunk in chunks:
        ...     print(f"Section: {chunk.section}, Length: {chunk.length}")
    """

    # Patterns for legal structure
    SECTION_PATTERNS = {
        "uk": re.compile(
            r"^[ \t]*(?:"
            r"(?:SECTION|PART|CHAPTER|SCHEDULE)\s+\d+[A-Z]?\b|"
            r"\d+\.\s+[A-Z]|"
            r"(?:DEFINITIONS?|INTERPRETATION|COMMENCEMENT|GENERAL)[ \t]*$"
            r")",
            re.MULTILINE | re.IGNORECASE,
        ),
        "us": re.compile(
            r"^[ \t]*(?:"
            r"(?:ARTICLE|SECTION)\s+[IVXLC\d]+\b|§\s*\d+|"
            r"\d+\.\d+\s+[A-Z]|"
            r"(?:DEFINITIONS?|RECITALS|WHEREAS)\b"
            r")",
            re.MULTILINE | re.IGNORECASE,
        ),
        "eu": re.compile(
            r"^[ \t]*(?:"
            r"(?:CHAPTER|TITLE|SECTION|PART)\s+[IVXLC\d]+\b|"
            r"Article\s+\d+[a-z]?\b"
            r")",
            re.MULTILINE | re.IGNORECASE,
        ),
    }

    # Paragraph and clause markers at the start of a line:
    # 1.1, 1.1.1, 12., (a), (aa), (iv), (2), a), 3)
    PARAGRAPH_PATTERN = re.compile(
        r"^[ \t]*(?:"
        r"\d+(?:\.\d+)+\.?|"
        r"\d+\.|"
        r"\((?:[a-z]{1,2}|[ivxlc]+|\d+)\)|"
        r"[a-z]\)|"
        r"\d+\)"
        r")(?=\s)",
        re.MULTILINE,
    )

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int | None = None,
        jurisdiction: str = "uk",
        respect_sections: bool = True,
        respect_paragraphs: bool = True,
        min_chunk_size: int | None = None,
    ):
        """
        Initialise the chunker.

        Args:
            chunk_size: Target chunk size in tokens (estimated at
                ~4 characters per token)
            overlap: Tokens repeated from the end of the previous chunk.
                Defaults to 10% of chunk_size, at most 50.
            jurisdiction: Jurisdiction for structure detection ('uk', 'us', 'eu')
            respect_sections: Try to keep sections together
            respect_paragraphs: Try to keep paragraphs together
            min_chunk_size: Chunks smaller than this many tokens are merged
                into a neighbour. Defaults to 20% of chunk_size, at most 100.

        Raises:
            ValueError: If the sizes are inconsistent.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap is None:
            overlap = min(50, chunk_size // 10)
        if not 0 <= overlap < chunk_size:
            raise ValueError(f"overlap must be between 0 and chunk_size - 1, got {overlap}")
        if min_chunk_size is None:
            min_chunk_size = min(100, chunk_size // 5)
        if not 0 <= min_chunk_size < chunk_size:
            raise ValueError(
                f"min_chunk_size must be between 0 and chunk_size - 1, got {min_chunk_size}"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.jurisdiction = jurisdiction.lower()
        self.respect_sections = respect_sections
        self.respect_paragraphs = respect_paragraphs
        self.min_chunk_size = min_chunk_size

        self.section_pattern = self.SECTION_PATTERNS.get(
            self.jurisdiction, self.SECTION_PATTERNS["uk"]
        )

    def chunk(self, text: str) -> list[str]:
        """
        Chunk text into training-ready pieces.

        Args:
            text: Legal document text

        Returns:
            List of text chunks
        """
        return [c.text for c in self.chunk_with_metadata(text)]

    def chunk_with_metadata(self, text: str) -> list[Chunk]:
        """
        Chunk text and return Chunk objects with metadata.

        Args:
            text: Legal document text

        Returns:
            List of Chunk objects with exact character offsets, the heading
            of the section each chunk starts in, and its first paragraph marker.
        """
        if not text.strip():
            return []

        # Leave room for the overlap so that every chunk, overlap included,
        # stays within chunk_size.
        limit = (self.chunk_size - self.overlap) * CHARS_PER_TOKEN
        spans = self._merge_small(self._pack(self._units(text, limit), limit), limit)

        chunks = []
        for index, (own_start, end, section) in enumerate(spans):
            start = own_start
            if index > 0 and self.overlap:
                start = self._overlap_start(text, own_start, spans[index - 1][0])
            start, end = _strip(text, start, end)
            if start >= end:
                continue
            chunks.append(
                Chunk(
                    text=text[start:end],
                    start_char=start,
                    end_char=end,
                    section=section,
                    paragraph=self._first_marker(text, own_start, end),
                )
            )
        return chunks

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        This is a rough estimate. For accurate counts,
        use the actual tokenizer of your target model.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // CHARS_PER_TOKEN

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _units(self, text: str, limit: int) -> list[tuple[int, int, str | None]]:
        """Split text into contiguous pieces no longer than limit, coarsest first."""
        if self.respect_sections:
            sections = self._section_spans(text)
        else:
            sections = [(None, 0, len(text))]

        units = []
        for name, start, end in sections:
            if end - start <= limit:
                units.append((start, end, name))
                continue
            if self.respect_paragraphs:
                paragraphs = self._paragraph_spans(text, start, end)
            else:
                paragraphs = [(start, end)]
            for p_start, p_end in paragraphs:
                if p_end - p_start <= limit:
                    units.append((p_start, p_end, name))
                else:
                    units.extend(
                        (s, e, name) for s, e in self._sentence_spans(text, p_start, p_end, limit)
                    )
        return units

    def _section_spans(self, text: str) -> list[tuple[str | None, int, int]]:
        """Sections as (heading, start, end), covering the whole text."""
        starts = [m.start() for m in self.section_pattern.finditer(text)]
        if not starts or starts[0] > 0:
            starts.insert(0, 0)
        spans = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            heading = None
            if self.section_pattern.match(text, start):
                line = text[start:end].strip().split("\n", 1)[0]
                heading = line if len(line) <= 80 else line[:77].rstrip() + "..."
            spans.append((heading, start, end))
        return spans

    def _paragraph_spans(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Split at blank lines and at paragraph or clause markers."""
        boundaries = set()
        for match in re.finditer(r"\n[ \t]*\n\s*", text[start:end]):
            # Split at the start of the line that follows the blank line(s).
            next_text = start + match.end()
            boundaries.add(text.rfind("\n", 0, next_text) + 1)
        boundaries.update(m.start() for m in self.PARAGRAPH_PATTERN.finditer(text, start, end))
        points = [start] + sorted(b for b in boundaries if start < b < end) + [end]
        return [(a, b) for a, b in zip(points, points[1:], strict=False) if b > a]

    def _sentence_spans(self, text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
        """Split at sentence boundaries; split over-long sentences at whitespace."""
        points = [start]
        for match in _SENTENCE_END.finditer(text, start, end):
            if not _is_abbreviation(text, match.start()):
                points.append(match.end())
        points.append(end)

        spans = []
        for a, b in zip(points, points[1:], strict=False):
            while b - a > limit:
                cut = _last_whitespace(text, a, a + limit)
                spans.append((a, cut))
                a = cut
            if b > a:
                spans.append((a, b))
        return spans

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------

    @staticmethod
    def _pack(
        units: list[tuple[int, int, str | None]], limit: int
    ) -> list[tuple[int, int, str | None]]:
        """Greedily join consecutive units into chunks of at most limit characters."""
        packed: list[tuple[int, int, str | None]] = []
        for start, end, section in units:
            if packed and end - packed[-1][0] <= limit:
                packed[-1] = (packed[-1][0], end, packed[-1][2])
            else:
                packed.append((start, end, section))
        return packed

    def _merge_small(
        self, spans: list[tuple[int, int, str | None]], limit: int
    ) -> list[tuple[int, int, str | None]]:
        """Merge chunks below min_chunk_size into a neighbour when that stays near the limit."""
        min_chars = self.min_chunk_size * CHARS_PER_TOKEN
        if not min_chars:
            return spans
        merged: list[tuple[int, int, str | None]] = []
        pending: tuple[int, int, str | None] | None = None
        for start, end, section in spans:
            if pending is not None:
                # A small chunk waiting to be merged into this one.
                if end - pending[0] <= limit + min_chars:
                    start, section = pending[0], pending[2]
                else:
                    merged.append(pending)
                pending = None
            if end - start < min_chars:
                if merged and end - merged[-1][0] <= limit + min_chars:
                    merged[-1] = (merged[-1][0], end, merged[-1][2])
                    continue
                pending = (start, end, section)
                continue
            merged.append((start, end, section))
        if pending is not None:
            merged.append(pending)
        return merged

    def _overlap_start(self, text: str, own_start: int, previous_start: int) -> int:
        """Start the chunk earlier by up to `overlap` tokens, on a word boundary."""
        target = max(previous_start, own_start - self.overlap * CHARS_PER_TOKEN)
        if target >= own_start:
            return own_start
        if target > 0 and not text[target - 1].isspace():
            match = re.compile(r"\s").search(text, target, own_start)
            if match is None:
                return own_start
            target = match.end()
        return target

    def _first_marker(self, text: str, start: int, end: int) -> str | None:
        """The first paragraph marker at the start of a line in text[start:end]."""
        line_start = text.rfind("\n", 0, start) + 1
        for match in self.PARAGRAPH_PATTERN.finditer(text, line_start, end):
            marker = match.group(0).strip()
            if match.end() - len(marker) >= start:
                return marker
        return None


def _is_abbreviation(text: str, period: int) -> bool:
    """True if the full stop at ``period`` ends an abbreviation or initial."""
    word_start = period
    while word_start > 0 and not text[word_start - 1].isspace():
        word_start -= 1
    word = text[word_start : period + 1].lstrip("([\"'“‘")
    if re.fullmatch(r"(?:[A-Za-z]\.)+", word):  # initials and "U.S."
        return True
    return word[:-1].lower() in _ABBREVIATIONS


def _last_whitespace(text: str, start: int, end: int) -> int:
    """Position after the last whitespace in text[start:end], or end if there is none."""
    for i in range(end, start + 1, -1):
        if text[i - 1].isspace():
            return i
    return end


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
