"""
Containers for legal training data.

A LegalDataset is a list of LegalSample objects. A sample is either a raw
document (for continued pre-training) or an instruction/response pair (for
instruction tuning), where ``text`` holds the supporting document.
"""

import hashlib
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from legalkit.data.formatting import to_instruction_format
from legalkit.preprocess.chunker import LegalChunker
from legalkit.preprocess.processor import LegalPreprocessor

logger = logging.getLogger(__name__)

# Field names accepted when reading records, in order of preference. They
# cover the common Alpaca, prompt/completion and question/answer layouts.
TEXT_KEYS = ("text", "input", "context", "document", "content", "body")
CONTEXT_KEYS = ("input", "context", "document", "content", "body", "text")
INSTRUCTION_KEYS = ("instruction", "prompt", "question", "query")
RESPONSE_KEYS = ("response", "output", "answer", "completion", "target")

TEXT_EXTENSIONS = (".txt", ".md")
RECORD_EXTENSIONS = (".jsonl", ".json")

# Folder names that imply a document type when loading a directory tree.
DOCUMENT_TYPE_FOLDERS = {
    "contract": "contract",
    "contracts": "contract",
    "agreements": "contract",
    "case": "case",
    "cases": "case",
    "judgment": "case",
    "judgments": "case",
    "judgement": "case",
    "judgements": "case",
    "legislation": "legislation",
    "statutes": "legislation",
    "regulations": "legislation",
}


@dataclass
class LegalSample:
    """
    One legal training example.

    Attributes:
        text: Document text, or the context for an instruction sample.
        instruction: Task description for instruction tuning.
        response: Target output for instruction tuning.
        document_type: e.g. "contract", "case", "legislation".
        jurisdiction: Jurisdiction code, e.g. "uk".
        source: Where the sample came from (file path, dataset id, URL).
        metadata: Any other fields carried through from the source record.
    """

    text: str = ""
    instruction: str | None = None
    response: str | None = None
    document_type: str = "unknown"
    jurisdiction: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_instruction(self) -> bool:
        """True when the sample has both an instruction and a response."""
        return bool(self.instruction) and self.response is not None

    def to_training_text(self, template: str = "alpaca") -> str:
        """Render the sample as training text (raw text or formatted instruction)."""
        if self.is_instruction:
            return to_instruction_format(
                self.instruction or "",
                self.response,
                context=self.text or None,
                template=template,
            )
        return self.text

    def to_messages(self, system: str | None = None) -> list[dict[str, str]]:
        """
        Render an instruction sample as chat messages.

        Raises:
            ValueError: If the sample is not an instruction sample.
        """
        if not self.is_instruction:
            raise ValueError("Only instruction samples can be converted to chat messages")
        user = f"{self.instruction}\n\n{self.text}" if self.text else str(self.instruction)
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": str(self.response)})
        return messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict, omitting empty optional fields."""
        record: dict[str, Any] = {"text": self.text}
        for key in ("instruction", "response", "jurisdiction", "source"):
            value = getattr(self, key)
            if value is not None:
                record[key] = value
        record["document_type"] = self.document_type
        if self.metadata:
            record["metadata"] = self.metadata
        return record

    @classmethod
    def from_dict(
        cls, record: dict[str, Any], text_key: str | None = None, **defaults: Any
    ) -> "LegalSample":
        """
        Build a sample from a record, accepting common field-name aliases.

        Unrecognised keys are kept in ``metadata``. ``defaults`` supplies
        document_type, jurisdiction or source when the record lacks them.

        Args:
            record: The source record.
            text_key: Field holding the text; detected when omitted.
            **defaults: document_type, jurisdiction or source.
        """
        record = dict(record)
        instruction = _pop_first(record, INSTRUCTION_KEYS)
        response = _pop_first(record, RESPONSE_KEYS)
        if text_key is not None:
            text = _as_text(record.pop(text_key, None))
        else:
            # In instruction data a "text" column is often the fully rendered
            # prompt and answer; the context is "input" (kept in metadata otherwise).
            text_keys = CONTEXT_KEYS if instruction is not None else TEXT_KEYS
            text = _as_text(_pop_first(record, text_keys))
        document_type = record.pop("document_type", None) or defaults.get("document_type")
        jurisdiction = record.pop("jurisdiction", None) or defaults.get("jurisdiction")
        source = record.pop("source", None) or defaults.get("source")
        metadata = record.pop("metadata", None) or {}
        return cls(
            text=text or "",
            instruction=None if instruction is None else _as_text(instruction),
            response=None if response is None else _as_text(response),
            document_type=document_type or "unknown",
            jurisdiction=jurisdiction,
            source=source,
            metadata={**record, **metadata},
        )


class LegalDataset:
    """
    A list of LegalSample objects with loading, cleaning and export helpers.

    Example:
        >>> dataset = LegalDataset.from_directory("./contracts")
        >>> dataset.preprocess(anonymise=True, jurisdiction="uk")
        >>> train, val, test = dataset.split(train=0.8, val=0.1, test=0.1)
        >>> train.to_jsonl("./train.jsonl")
    """

    def __init__(self, samples: Iterable[LegalSample] | None = None):
        self.samples: list[LegalSample] = list(samples) if samples is not None else []

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[LegalSample]:
        return iter(self.samples)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return LegalDataset(self.samples[index])
        return self.samples[index]

    def __repr__(self) -> str:
        return f"LegalDataset({len(self.samples)} samples)"

    def append(self, sample: LegalSample) -> None:
        """Add one sample."""
        self.samples.append(sample)

    def extend(self, samples: Iterable[LegalSample]) -> None:
        """Add several samples."""
        self.samples.extend(samples)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(cls, texts: Iterable[str], **defaults: Any) -> "LegalDataset":
        """Create a dataset with one sample per text."""
        return cls(LegalSample.from_dict({"text": text}, **defaults) for text in texts)

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]], **defaults: Any) -> "LegalDataset":
        """Create a dataset from dict records (see LegalSample.from_dict)."""
        return cls(LegalSample.from_dict(record, **defaults) for record in records)

    @classmethod
    def from_jsonl(cls, path: str | Path, **defaults: Any) -> "LegalDataset":
        """
        Load a JSON Lines file, one record per line.

        Records without a source are labelled with the file and line number,
        e.g. "cases.jsonl#12".

        Raises:
            ValueError: If a line is not a JSON object.
        """
        path = Path(path)
        return cls._from_numbered_records(read_jsonl(path), path.name, defaults)

    @classmethod
    def from_json(cls, path: str | Path, **defaults: Any) -> "LegalDataset":
        """
        Load a JSON file holding a list of records, or {"data": [...]}.

        Records without a source are labelled with the file and position,
        e.g. "cases.json#3".
        """
        path = Path(path)
        return cls._from_numbered_records(read_json(path), path.name, defaults)

    @classmethod
    def _from_numbered_records(
        cls, records: Iterable[tuple[int, dict[str, Any]]], label: str, defaults: dict[str, Any]
    ) -> "LegalDataset":
        return cls(
            LegalSample.from_dict(record, **{"source": f"{label}#{number}", **defaults})
            for number, record in records
        )

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        extensions: Iterable[str] = TEXT_EXTENSIONS + RECORD_EXTENSIONS,
        recursive: bool = True,
        encoding: str = "utf-8",
        **defaults: Any,
    ) -> "LegalDataset":
        """
        Load every supported file in a directory, in sorted order.

        Text files (.txt, .md) become one sample each. JSON and JSONL files are
        read as records. Hidden files and folders are skipped. When no
        document_type is given, it is inferred from folder names such as
        ``contracts/`` or ``judgments/``.

        Args:
            path: Directory to load.
            extensions: File extensions to include.
            recursive: Also load files in subdirectories.
            encoding: Encoding of text files.
            **defaults: document_type, jurisdiction or source for every sample.
        """
        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        wanted = {ext.lower() for ext in extensions}
        files = sorted(root.rglob("*") if recursive else root.glob("*"))
        dataset = cls()
        for file in files:
            relative = file.relative_to(root)
            if not file.is_file() or any(part.startswith(".") for part in relative.parts):
                continue
            suffix = file.suffix.lower()
            if suffix not in wanted:
                continue

            file_defaults = dict(defaults)
            if "document_type" not in defaults:
                inferred = _document_type_from_path(relative)
                if inferred:
                    file_defaults["document_type"] = inferred

            label = relative.as_posix()
            if suffix in TEXT_EXTENSIONS:
                text = file.read_text(encoding=encoding)
                record = {"text": text}
                dataset.append(LegalSample.from_dict(record, **{"source": label, **file_defaults}))
            elif suffix == ".jsonl":
                dataset.extend(cls._from_numbered_records(read_jsonl(file), label, file_defaults))
            elif suffix == ".json":
                dataset.extend(cls._from_numbered_records(read_json(file), label, file_defaults))
        return dataset

    @classmethod
    def from_huggingface(
        cls,
        hf_dataset: Iterable[dict[str, Any]],
        text_field: str | None = None,
        limit: int | None = None,
        **defaults: Any,
    ) -> "LegalDataset":
        """
        Convert a Hugging Face dataset (regular or streaming) to a LegalDataset.

        Args:
            hf_dataset: A ``datasets.Dataset`` or ``IterableDataset``.
            text_field: Column holding the document text. Detected when omitted.
            limit: Stop after this many rows (useful with streaming corpora).
            **defaults: document_type, jurisdiction or source for every sample.
        """
        dataset = cls()
        skipped = 0
        for index, row in enumerate(hf_dataset):
            if limit is not None and len(dataset) >= limit:
                break
            record = dict(row)
            if text_field is not None and text_field not in record:
                raise KeyError(f"Column {text_field!r} not found. Available: {sorted(record)}")
            if text_field is None and index == 0:
                if not any(k in record for k in TEXT_KEYS + INSTRUCTION_KEYS):
                    raise ValueError(
                        f"Could not find a text column. Pass text_field= one of {sorted(record)}"
                    )
            sample = LegalSample.from_dict(record, text_key=text_field, **defaults)
            if not sample.text and not sample.instruction:
                skipped += 1
                continue
            dataset.append(sample)
        if skipped:
            logger.warning(f"Skipped {skipped} rows with no text")
        return dataset

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        """Return every sample as a dict."""
        return [sample.to_dict() for sample in self.samples]

    def to_jsonl(self, path: str | Path) -> Path:
        """Write the dataset as JSON Lines, creating parent folders. Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in self.to_records():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def to_huggingface(self, template: str = "alpaca", text_column: str = "text"):
        """
        Convert to a ``datasets.Dataset`` ready for supervised fine-tuning.

        Args:
            template: Prompt template for instruction samples (see
                to_instruction_format). Use "messages" to produce a
                conversational "messages" column that the trainer formats
                with the model's own chat template.
            text_column: Name of the text column (ignored for "messages").
        """
        try:
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(
                "The 'datasets' package is required: pip install 'legal-llm-toolkit[train]'"
            ) from e

        if template == "messages":
            return Dataset.from_dict({"messages": [s.to_messages() for s in self.samples]})
        return Dataset.from_dict(
            {text_column: [s.to_training_text(template) for s in self.samples]}
        )

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def preprocess(
        self,
        anonymise: bool = False,
        jurisdiction: str = "uk",
        normalise_citations: bool = True,
        preserve_case_names: bool = True,
        **processor_kwargs: Any,
    ) -> "LegalDataset":
        """
        Clean every sample in place with LegalPreprocessor.

        When anonymising, the instruction, response and text values in
        metadata are anonymised with the same mapping as the text, so
        placeholders stay consistent within each sample. The PII mapping
        itself is never stored in the dataset.

        Args:
            anonymise: Replace personal data with placeholders.
            jurisdiction: Jurisdiction for citation parsing.
            normalise_citations: Standardise case citation formatting.
            preserve_case_names: Keep party names inside case citations.
            **processor_kwargs: Other LegalPreprocessor options.

        Returns:
            This dataset, to allow chaining.
        """
        processor = LegalPreprocessor(
            jurisdiction=jurisdiction,
            anonymise=anonymise,
            normalise_citations=normalise_citations,
            preserve_case_names=preserve_case_names,
            **processor_kwargs,
        )
        for sample in self.samples:
            reset = True
            if sample.text:
                result = processor.process(sample.text)
                sample.text = result.processed
                sample.metadata["citation_count"] = len(result.citations)
                reset = False
            if processor.anonymiser is not None:
                anonymiser = processor.anonymiser
                for attr in ("instruction", "response"):
                    value = getattr(sample, attr)
                    if value:
                        setattr(sample, attr, anonymiser.anonymise(value, reset=reset).text)
                        reset = False
                # Metadata can hold personal data too (e.g. a rendered prompt).
                for key, value in sample.metadata.items():
                    if isinstance(value, str) and value:
                        sample.metadata[key] = anonymiser.anonymise(value, reset=reset).text
                        reset = False
                    elif (
                        isinstance(value, list) and value and all(isinstance(v, str) for v in value)
                    ):
                        sample.metadata[key] = [
                            anonymiser.anonymise(v, reset=False).text for v in value
                        ]
                sample.metadata["anonymised"] = True
            if sample.jurisdiction is None:
                sample.jurisdiction = jurisdiction
        return self

    def chunk(
        self,
        chunk_size: int = 512,
        overlap: int | None = None,
        jurisdiction: str = "uk",
        min_chunk_size: int | None = None,
    ) -> "LegalDataset":
        """
        Split long documents into chunks, keeping legal structure together.

        Instruction samples are passed through unchanged. Each chunk records
        its position in ``metadata`` (chunk_index, chunk_count, start_char,
        end_char, section) and the document it came from (document_id), so
        ``split(group_by="document_id")`` keeps every chunk of a document in
        the same split.

        Returns:
            A new LegalDataset.
        """
        chunker = LegalChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            jurisdiction=jurisdiction,
            min_chunk_size=min_chunk_size,
        )
        chunked = LegalDataset()
        for position, sample in enumerate(self.samples):
            if sample.is_instruction or not sample.text:
                chunked.append(sample)
                continue
            document_id = sample.metadata.get("document_id")
            if document_id is None:
                document_id = f"{sample.source or ''}#{position}"
            pieces = chunker.chunk_with_metadata(sample.text)
            for index, piece in enumerate(pieces):
                metadata = {
                    **sample.metadata,
                    "document_id": document_id,
                    "chunk_index": index,
                    "chunk_count": len(pieces),
                    "start_char": piece.start_char,
                    "end_char": piece.end_char,
                }
                if piece.section:
                    metadata["section"] = piece.section
                chunked.append(replace(sample, text=piece.text, metadata=metadata))
        return chunked

    def filter(self, predicate: Callable[[LegalSample], bool]) -> "LegalDataset":
        """Return a new dataset with the samples for which predicate is true."""
        return LegalDataset(s for s in self.samples if predicate(s))

    def map(self, fn: Callable[[LegalSample], LegalSample]) -> "LegalDataset":
        """Return a new dataset with fn applied to every sample."""
        return LegalDataset(fn(s) for s in self.samples)

    def deduplicate(self) -> "LegalDataset":
        """
        Drop exact duplicates, ignoring case and whitespace differences.

        The same judgment or contract often appears in several sources;
        duplicates overweight it in training and leak between splits.
        """
        seen: set[str] = set()
        unique = LegalDataset()
        for sample in self.samples:
            key = _fingerprint(sample)
            if key not in seen:
                seen.add(key)
                unique.append(sample)
        return unique

    def shuffle(self, seed: int | None = 42) -> "LegalDataset":
        """Return a new dataset in random order."""
        samples = list(self.samples)
        random.Random(seed).shuffle(samples)
        return LegalDataset(samples)

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int | None = 42,
        shuffle: bool = True,
        group_by: str | None = None,
    ) -> tuple["LegalDataset", "LegalDataset", "LegalDataset"]:
        """
        Split into train, validation and test sets.

        Args:
            train: Fraction for training.
            val: Fraction for validation.
            test: Fraction for testing.
            seed: Random seed for reproducible splits.
            shuffle: Shuffle before splitting.
            group_by: Keep samples that share this attribute or metadata key
                in the same split, e.g. "source" so chunks of one judgment
                cannot leak from training into the test set.

        Returns:
            Tuple of (train, val, test) datasets.
        """
        fractions = {"train": train, "val": val, "test": test}
        for name, fraction in fractions.items():
            if fraction < 0:
                raise ValueError(f"{name} fraction must not be negative")
        if abs(sum(fractions.values()) - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {sum(fractions.values())}")

        groups: dict[Any, list[LegalSample]] = {}
        for index, sample in enumerate(self.samples):
            key = _group_key(sample, group_by) if group_by else None
            # Samples without a group key are split individually.
            groups.setdefault(("sample", index) if key is None else ("group", key), []).append(
                sample
            )
        units = list(groups.values())
        if shuffle:
            random.Random(seed).shuffle(units)

        # Give each group to the split furthest below its target size (ties
        # go to train, then val), so large groups cannot starve the others.
        total = len(self.samples)
        targets = [total * fraction for fraction in (train, val, test)]
        open_splits = [i for i, fraction in enumerate((train, val, test)) if fraction > 0]
        splits: list[list[LegalSample]] = [[], [], []]
        for unit in units:
            best = max(open_splits, key=lambda i: (targets[i] - len(splits[i]), -i))
            splits[best].extend(unit)
        return LegalDataset(splits[0]), LegalDataset(splits[1]), LegalDataset(splits[2])

    def statistics(self) -> dict[str, Any]:
        """Summary statistics: sizes, document types and jurisdictions."""
        lengths = [
            len(s.text) + len(s.instruction or "") + len(s.response or "") for s in self.samples
        ]
        total = sum(lengths)
        return {
            "num_samples": len(self.samples),
            "num_instruction_samples": sum(1 for s in self.samples if s.is_instruction),
            "total_chars": total,
            "avg_chars": round(total / len(lengths), 1) if lengths else 0,
            "min_chars": min(lengths, default=0),
            "max_chars": max(lengths, default=0),
            "estimated_tokens": total // 4,
            "document_types": dict(Counter(s.document_type for s in self.samples)),
            "jurisdictions": dict(Counter(s.jurisdiction or "unknown" for s in self.samples)),
        }


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Records of a JSON Lines file with their line numbers."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({e.msg})") from e
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append((line_number, record))
    return records


def read_json(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Records of a JSON file (a list, or {"data": [...]}) numbered from 1."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise ValueError(f"{path}: expected a list of JSON objects")
    return list(enumerate(data, start=1))


def _pop_first(record: dict[str, Any], keys: Iterable[str]) -> Any:
    """Pop and return the first present, non-None key from record."""
    for key in keys:
        if record.get(key) is not None:
            return record.pop(key)
    return None


def _as_text(value: Any) -> str:
    """Coerce a field value to text; lists of paragraphs are joined."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return "\n\n".join(value)
    return str(value)


def _document_type_from_path(relative: Path) -> str | None:
    for part in reversed(relative.parts[:-1]):
        document_type = DOCUMENT_TYPE_FOLDERS.get(part.lower())
        if document_type:
            return document_type
    return None


def _fingerprint(sample: LegalSample) -> str:
    parts = (sample.instruction or "", sample.text, sample.response or "")
    normalised = "\x1f".join(re.sub(r"\s+", " ", p).strip().lower() for p in parts)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _group_key(sample: LegalSample, key: str) -> Any:
    value = getattr(sample, key, None)
    if value is None:
        value = sample.metadata.get(key)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value
