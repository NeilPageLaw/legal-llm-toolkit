"""
Loaders for local legal corpora and Hugging Face datasets.
"""

from pathlib import Path
from typing import Any

from legalkit.data.dataset import LegalDataset, LegalSample

LOCAL_SUFFIXES = (".jsonl", ".json", ".txt", ".md")


def load_legal_corpus(
    source: str | Path,
    *,
    split: str = "train",
    name: str | None = None,
    text_field: str | None = None,
    limit: int | None = None,
    streaming: bool = False,
    document_type: str | None = None,
    jurisdiction: str | None = None,
    **load_kwargs: Any,
) -> LegalDataset:
    """
    Load a legal corpus from a local path or the Hugging Face Hub.

    Local sources can be a directory, a .jsonl/.json file of records, or a
    single .txt/.md document. Anything else is treated as a Hugging Face
    dataset id and loaded with ``datasets.load_dataset``.

    Args:
        source: Local path or Hugging Face dataset id.
        split: Dataset split to load (Hugging Face only).
        name: Dataset configuration name (Hugging Face only).
        text_field: Column holding the document text. Detected when omitted.
        limit: Maximum number of samples to load.
        streaming: Stream rows instead of downloading the whole dataset;
            combine with ``limit`` for very large corpora.
        document_type: Document type recorded on each sample.
        jurisdiction: Jurisdiction recorded on each sample.
        **load_kwargs: Passed to ``datasets.load_dataset``.

    Returns:
        LegalDataset

    Example:
        >>> corpus = load_legal_corpus("./my_contracts/", jurisdiction="uk")
        >>> ledgar = load_legal_corpus("coastalcph/lex_glue", name="ledgar", limit=1000)
    """
    defaults: dict[str, Any] = {
        key: value
        for key, value in {"document_type": document_type, "jurisdiction": jurisdiction}.items()
        if value is not None
    }

    path = Path(source)
    if _exists(path):
        if path.is_dir():
            dataset = LegalDataset.from_directory(path, **defaults)
        elif path.suffix.lower() == ".jsonl":
            dataset = LegalDataset.from_jsonl(path, **defaults)
        elif path.suffix.lower() == ".json":
            dataset = LegalDataset.from_json(path, **defaults)
        elif path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
            dataset = LegalDataset(
                [LegalSample.from_dict({"text": text, "source": path.name}, **defaults)]
            )
        else:
            raise ValueError(
                f"Unsupported file type: {path.suffix} (use {', '.join(LOCAL_SUFFIXES)})"
            )
        return dataset[:limit] if limit is not None else dataset

    if path.suffix.lower() in LOCAL_SUFFIXES or str(source).startswith((".", "/", "~")):
        raise FileNotFoundError(f"No such file or directory: {source}")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f"{source!r} is not a local path. Loading it from the Hugging Face Hub "
            "needs the 'datasets' package: pip install 'legal-llm-toolkit[train]'"
        ) from e

    hf_dataset = load_dataset(str(source), name, split=split, streaming=streaming, **load_kwargs)
    return LegalDataset.from_huggingface(
        hf_dataset,
        text_field=text_field,
        limit=limit,
        source=str(source),
        **defaults,
    )


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:  # e.g. a string too long to be a file name
        return False
