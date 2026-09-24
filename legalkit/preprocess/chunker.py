"""
Intelligent chunking for legal documents.

Preserves legal structure (sections, paragraphs, clauses) while
creating appropriately sized chunks for LLM training.
"""

import re
from dataclasses import dataclass


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

    Example:
        >>> chunker = LegalChunker(chunk_size=512, overlap=50)
        >>> chunks = chunker.chunk(contract_text)
        >>> for chunk in chunks:
        ...     print(f"Section: {chunk.section}, Length: {chunk.length}")
    """

    # Patterns for legal structure
    SECTION_PATTERNS = {
        "uk": re.compile(
            r"^(?:"
            r"(?:SECTION|PART|CHAPTER|SCHEDULE)\s+\d+|"
            r"\d+\.\s+[A-Z]|"
            r"(?:DEFINITIONS?|INTERPRETATION|COMMENCEMENT|GENERAL)\s*$"
            r")",
            re.MULTILINE | re.IGNORECASE,
        ),
        "us": re.compile(
            r"^(?:"
            r"(?:ARTICLE|SECTION|§)\s+\d+|"
            r"\d+\.\d+\s+[A-Z]|"
            r"(?:DEFINITIONS?|RECITALS|WHEREAS)\s*$"
            r")",
            re.MULTILINE | re.IGNORECASE,
        ),
    }

    PARAGRAPH_PATTERN = re.compile(
        r"^(?:"
        r"\d+\.\d+(?:\.\d+)?|"  # 1.1 or 1.1.1
        r"\([a-z]\)|"  # (a)
        r"\([ivx]+\)|"  # (i), (ii), (iii)
        r"[a-z]\)|"  # a)
        r"\d+\)"  # 1)
        r")\s+",
        re.MULTILINE,
    )

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        jurisdiction: str = "uk",
        respect_sections: bool = True,
        respect_paragraphs: bool = True,
        min_chunk_size: int = 100,
    ):
        """
        Initialise the chunker.

        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Number of tokens to overlap between chunks
            jurisdiction: Jurisdiction for structure detection
            respect_sections: Try to keep sections together
            respect_paragraphs: Try to keep paragraphs together
            min_chunk_size: Minimum chunk size (won't split below this)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.jurisdiction = jurisdiction.lower()
        self.respect_sections = respect_sections
        self.respect_paragraphs = respect_paragraphs
        self.min_chunk_size = min_chunk_size

        self.section_pattern = self.SECTION_PATTERNS.get(jurisdiction, self.SECTION_PATTERNS["uk"])

    def chunk(self, text: str) -> list[str]:
        """
        Chunk text into training-ready pieces.

        Args:
            text: Legal document text

        Returns:
            List of text chunks
        """
        # First, identify structure
        sections = self._split_sections(text) if self.respect_sections else [text]

        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)

        return chunks

    def chunk_with_metadata(self, text: str) -> list[Chunk]:
        """
        Chunk text and return Chunk objects with metadata.

        Args:
            text: Legal document text

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_pos = 0

        sections = self._identify_sections(text)

        for section_name, section_text, section_start in sections:
            section_chunks = self._chunk_section(section_text)

            for chunk_text in section_chunks:
                # Find actual position in original text
                chunk_start = text.find(chunk_text, current_pos)
                if chunk_start == -1:
                    chunk_start = current_pos

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_text),
                        section=section_name,
                    )
                )

                current_pos = chunk_start + len(chunk_text) - self.overlap

        return chunks

    def _split_sections(self, text: str) -> list[str]:
        """Split text into major sections."""
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            return [text]

        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(text[start:end])

        # Add any text before first section
        if matches[0].start() > 0:
            sections.insert(0, text[: matches[0].start()])

        return sections

    def _identify_sections(self, text: str) -> list[tuple[str | None, str, int]]:
        """Identify sections with their names and positions."""
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            return [(None, text, 0)]

        sections = []

        # Text before first section
        if matches[0].start() > 0:
            sections.append((None, text[: matches[0].start()], 0))

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_name = match.group(0).strip()
            sections.append((section_name, text[start:end], start))

        return sections

    def _chunk_section(self, text: str) -> list[str]:
        """Chunk a single section."""
        # Rough token estimate (chars / 4)
        char_limit = self.chunk_size * 4
        overlap_chars = self.overlap * 4

        if len(text) <= char_limit:
            return [text.strip()] if text.strip() else []

        chunks = []

        if self.respect_paragraphs:
            # Split by paragraphs first
            paragraphs = self._split_paragraphs(text)
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) <= char_limit:
                    current_chunk += para
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    # If paragraph itself is too long, split it
                    if len(para) > char_limit:
                        para_chunks = self._split_long_text(para, char_limit, overlap_chars)
                        chunks.extend(para_chunks[:-1])
                        current_chunk = para_chunks[-1] if para_chunks else ""
                    else:
                        # Add overlap from previous chunk
                        if chunks:
                            overlap_text = chunks[-1][-overlap_chars:]
                            current_chunk = overlap_text + para
                        else:
                            current_chunk = para

            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            chunks = self._split_long_text(text, char_limit, overlap_chars)

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs, preserving paragraph markers."""
        # Split on double newlines or paragraph numbers
        parts = re.split(r"(\n\n+)", text)

        paragraphs = []
        current = ""

        for part in parts:
            if re.match(r"\n\n+", part):
                if current:
                    paragraphs.append(current)
                    current = ""
            else:
                # Check for paragraph markers within the text
                sub_parts = self.PARAGRAPH_PATTERN.split(part)
                if len(sub_parts) > 1:
                    for i, sub in enumerate(sub_parts):
                        if sub.strip():
                            if i > 0:
                                # This is after a paragraph marker
                                if current:
                                    paragraphs.append(current)
                                current = sub
                            else:
                                current += sub
                else:
                    current += part

        if current:
            paragraphs.append(current)

        return paragraphs

    def _split_long_text(self, text: str, char_limit: int, overlap_chars: int) -> list[str]:
        """Split long text at sentence boundaries."""
        # Try to split at sentence boundaries
        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_pattern.split(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= char_limit:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                if chunks and overlap_chars > 0:
                    overlap = chunks[-1][-overlap_chars:]
                    current_chunk = overlap + sentence + " "
                else:
                    current_chunk = sentence + " "

                # If single sentence is too long, force split
                while len(current_chunk) > char_limit:
                    chunks.append(current_chunk[:char_limit].strip())
                    current_chunk = current_chunk[char_limit - overlap_chars :]

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

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
        # Rough estimate: ~4 chars per token for English
        return len(text) // 4
