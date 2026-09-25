"""
Legal LLM Toolkit CLI.

Command-line interface for common toolkit operations.
"""

import argparse
import importlib.util
import json
import logging
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import legalkit
from legalkit._install import REPOSITORY_URL, install_command
from legalkit.eval.benchmark import LegalBenchmark
from legalkit.finetune.config import SUPPORTED_METHODS, SUPPORTED_TASKS
from legalkit.preprocess.anonymiser import EntityType
from legalkit.preprocess.citations import SUPPORTED_JURISDICTIONS

JURISDICTIONS = SUPPORTED_JURISDICTIONS
TRAINING_METHODS = SUPPORTED_METHODS
TRAINING_TASKS = SUPPORTED_TASKS
BENCHMARK_TASKS = tuple(LegalBenchmark.AVAILABLE_TASKS)
ENTITY_TYPES = tuple(entity_type.value for entity_type in EntityType)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the legalkit command."""
    parser = argparse.ArgumentParser(
        prog="legalkit",
        description="Legal LLM Toolkit - Fine-tune and evaluate LLMs on legal text",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {legalkit.__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show progress logs and full error tracebacks"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Preprocess command
    preprocess = subparsers.add_parser(
        "preprocess",
        help="Clean, anonymise and chunk legal documents",
        description="Clean, anonymise and chunk a document or a dataset. A single text "
        "file is written as text (or JSON Lines with --chunk); a directory or "
        ".jsonl/.json file is written as JSON Lines, or summarised without -o.",
    )
    preprocess.add_argument(
        "input", help="A text file, a .jsonl/.json file of records, or a directory"
    )
    preprocess.add_argument("-o", "--output", help="Output file (default: standard output)")
    _add_jurisdiction(preprocess)
    preprocess.add_argument(
        "--anonymise", action="store_true", help="Replace personal data with placeholders"
    )
    _add_anonymiser_options(preprocess)
    preprocess.add_argument(
        "--chunk", action="store_true", help="Split documents into training chunks"
    )
    preprocess.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size in tokens (default: 512)"
    )
    preprocess.add_argument(
        "--chunk-overlap",
        type=int,
        help="Tokens repeated between chunks (default: 10%% of the chunk size, at most 50)",
    )
    preprocess.add_argument(
        "--no-normalise-citations",
        action="store_true",
        help="Leave case citations as written instead of standardising them",
    )
    preprocess.add_argument("--dedupe", action="store_true", help="Drop duplicate documents")
    preprocess.set_defaults(handler=cmd_preprocess)

    # Anonymise command
    anonymise = subparsers.add_parser(
        "anonymise",
        aliases=["anonymize"],
        help="Anonymise a document",
        description="Replace personal data in a document with placeholders such as "
        "[PERSON_1]. Rule-based detection misses untitled names unless --ner is "
        "used: always review the output.",
    )
    anonymise.add_argument("input", help="A text file, or - for standard input")
    anonymise.add_argument("-o", "--output", help="Output file (default: standard output)")
    anonymise.add_argument(
        "--mapping",
        help="Also write the placeholder mapping to this JSON file. It contains the "
        "original personal data: store it securely",
    )
    _add_anonymiser_options(anonymise)
    anonymise.set_defaults(handler=cmd_anonymise)

    # Citations command
    cite = subparsers.add_parser("citations", help="Extract citations from text")
    cite.add_argument("input", help="A file, - for standard input, or the text itself")
    _add_jurisdiction(cite)
    cite.add_argument("--json", action="store_true", help="Output as JSON")
    cite.set_defaults(handler=cmd_citations)

    # Train command
    train = subparsers.add_parser(
        "train",
        help="Fine-tune a legal LLM",
        description="Fine-tune a model with LoRA, QLoRA or full fine-tuning. Options "
        "override settings loaded with --config.",
    )
    train.add_argument(
        "dataset",
        help="Training data: a directory, a .jsonl/.json file or a Hugging Face dataset id",
    )
    train.add_argument("-m", "--model", help="Base model (default: mistralai/Mistral-7B-v0.1)")
    train.add_argument(
        "--method", choices=TRAINING_METHODS, help="Fine-tuning method (default: qlora)"
    )
    train.add_argument(
        "--task", choices=TRAINING_TASKS, help="Legal task preset (default: general)"
    )
    train.add_argument("-o", "--output", help="Output directory (default: ./legal-llm-output)")
    train.add_argument("--epochs", type=int, help="Number of epochs (default: set by the task)")
    train.add_argument("--batch-size", type=int, help="Batch size (default: 4)")
    train.add_argument("--learning-rate", type=float, help="Learning rate (default: 2e-4)")
    train.add_argument(
        "--max-seq-length", type=int, help="Maximum tokens per example (default: set by the task)"
    )
    train.add_argument(
        "-j", "--jurisdiction", choices=JURISDICTIONS, help="Jurisdiction (default: uk)"
    )
    train.add_argument("--eval-dataset", help="Evaluation data, in the same forms as the dataset")
    train.add_argument("--anonymise", action="store_true", help="Anonymise the training data first")
    train.add_argument(
        "--template", help="Prompt template: alpaca, chatml or messages (default: alpaca)"
    )
    train.add_argument("--config", help="JSON file of LegalTrainingConfig settings")
    train.add_argument(
        "--merge",
        action="store_true",
        help="Save a standalone merged model instead of a LoRA adapter",
    )
    train.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow the model repository to run its own code. Only for models you trust",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the configuration and a memory estimate, then exit",
    )
    train.set_defaults(handler=cmd_train)

    # Evaluate command
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a legal LLM")
    evaluate.add_argument("model", help="Model path or Hugging Face id")
    evaluate.add_argument(
        "--tasks", nargs="+", choices=BENCHMARK_TASKS, help="Tasks to run (default: all)"
    )
    evaluate.add_argument(
        "--data",
        help="Directory of <task>.jsonl test files (default: the built-in samples, a smoke test only)",
    )
    evaluate.add_argument("-o", "--output", help="Save results, including model outputs, as JSON")
    evaluate.add_argument("--max-samples", type=int, help="Maximum samples per task")
    evaluate.add_argument(
        "--max-new-tokens", type=int, default=512, help="Generation length limit (default: 512)"
    )
    evaluate.add_argument("--template", help="Prompt template used for fine-tuning, e.g. alpaca")
    _add_jurisdiction(evaluate)
    evaluate.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow the model repository to run its own code. Only for models you trust",
    )
    evaluate.set_defaults(handler=cmd_evaluate)

    # Info command
    info = subparsers.add_parser("info", help="Show toolkit information")
    info.set_defaults(handler=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 0

    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        if args.verbose:
            traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_preprocess(args) -> int:
    """Handle preprocess command."""
    from legalkit.data import load_legal_corpus
    from legalkit.preprocess import LegalChunker, LegalPreprocessor

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")
    anonymiser = _build_anonymiser(args) if args.anonymise or args.salt else None
    normalise_citations = not args.no_normalise_citations

    if path.is_file() and path.suffix.lower() not in (".jsonl", ".json"):
        processor = LegalPreprocessor(
            jurisdiction=args.jurisdiction,
            normalise_citations=normalise_citations,
            anonymiser=anonymiser,
        )
        result = processor.process(path.read_text(encoding="utf-8"))
        if args.chunk:
            chunker = LegalChunker(
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
                jurisdiction=args.jurisdiction,
            )
            records = [
                {
                    "text": chunk.text,
                    "source": path.name,
                    "chunk_index": index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "section": chunk.section,
                }
                for index, chunk in enumerate(chunker.chunk_with_metadata(result.processed))
            ]
            _write_output(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), args.output
            )
        else:
            _write_output(result.processed + "\n", args.output)

        if result.citations:
            print(f"Found {len(result.citations)} citations:", file=sys.stderr)
            for c in result.citations:
                print(f"  - {c.normalised or c.raw}", file=sys.stderr)
        if result.anonymisation is not None:
            _report_entities(result.anonymisation.entities)
        return 0

    dataset = load_legal_corpus(path)
    if args.dedupe:
        before = len(dataset)
        dataset = dataset.deduplicate()
        print(f"Removed {before - len(dataset)} duplicates", file=sys.stderr)
    dataset.preprocess(
        anonymise=anonymiser is not None,
        jurisdiction=args.jurisdiction,
        normalise_citations=normalise_citations,
        anonymiser=anonymiser,
    )
    if args.chunk:
        dataset = dataset.chunk(
            chunk_size=args.chunk_size, overlap=args.chunk_overlap, jurisdiction=args.jurisdiction
        )

    if args.output:
        dataset.to_jsonl(args.output)
        print(f"Saved {len(dataset)} samples to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(dataset.statistics(), indent=2))
    return 0


def cmd_anonymise(args) -> int:
    """Handle anonymise command."""
    text = _read_input(args.input, allow_literal=False)
    result = _build_anonymiser(args).anonymise(text)
    _write_output(result.text, args.output)
    _report_entities(result.entities)
    if args.mapping:
        path = Path(args.mapping)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Mapping written to {path}. It contains the original personal data: store it securely.",
            file=sys.stderr,
        )
    return 0


def cmd_citations(args) -> int:
    """Handle citations command."""
    from legalkit.preprocess import CitationParser

    text = _read_input(args.input, allow_literal=True)
    citations = CitationParser(jurisdiction=args.jurisdiction).parse(text)

    if args.json:
        print(json.dumps([c.to_dict() for c in citations], indent=2, ensure_ascii=False))
    elif citations:
        print(f"Found {len(citations)} citations:\n")
        for c in citations:
            print(f"  [{c.citation_type}] {c.normalised or c.raw}")
            if c.parties:
                print(f"           Case: {c.parties}")
            if c.court:
                print(f"           Court: {c.court}")
            if c.year:
                print(f"           Year: {c.year}")
            if c.paragraph:
                print(f"           Paragraph: {c.paragraph}")
            print()
    else:
        print("No citations found.")
    return 0


def cmd_train(args) -> int:
    """Handle train command."""
    from dataclasses import fields

    from legalkit.finetune import LegalTrainer, LegalTrainingConfig, estimate_memory_usage

    settings: dict[str, Any] = {}
    if args.config:
        settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
        unknown = sorted(
            set(settings)
            - {f.name for f in fields(LegalTrainingConfig) if f.init}
            - {"derived_settings", "derived_from"}
        )
        if unknown:
            raise ValueError(f"Unknown settings in {args.config}: {', '.join(unknown)}")
        # Values derived from the saved task, method or model (and not edited
        # since) are derived again, so options such as --method get matching ones.
        settings = LegalTrainingConfig.explicit_settings(settings)

    options = {
        "base_model": args.model,
        "method": args.method,
        "task": args.task,
        "output_dir": args.output,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "jurisdiction": args.jurisdiction,
        "prompt_template": args.template,
    }
    settings.update({k: v for k, v in options.items() if v is not None})
    if args.anonymise:
        settings["anonymise_training_data"] = True
    if args.trust_remote_code:
        settings["trust_remote_code"] = True
    config = LegalTrainingConfig.from_dict(settings)

    print("Training configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    estimate = estimate_memory_usage(
        config.base_model, config.method, config.batch_size, config.max_seq_length or 2048
    )
    if "error" not in estimate:
        print(
            f"\nEstimated GPU memory: ~{estimate['total_estimated_gb']} GB "
            f"({estimate['recommended_gpu']})"
        )
    if args.dry_run:
        return 0

    trainer = LegalTrainer(config)
    print(f"\nStarting training on {args.dataset}...")
    result = trainer.train(args.dataset, eval_dataset=args.eval_dataset)
    print("\nTraining complete!")
    print(f"Final loss: {result['train_loss']:.4f}")

    trainer.save(config.output_dir, merge_adapter=args.merge)
    print(f"Model saved to {config.output_dir}")
    return 0


def cmd_evaluate(args) -> int:
    """Handle evaluate command."""
    from legalkit.eval import LegalBenchmark

    benchmark = LegalBenchmark(
        tasks=args.tasks,
        jurisdiction=args.jurisdiction,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        prompt_template=args.template,
        trust_remote_code=args.trust_remote_code,
    )
    if not args.data:
        print(
            "No --data given: running the built-in samples. This is a smoke test, not a benchmark.",
            file=sys.stderr,
        )

    print(f"Running evaluation on {args.model}...", file=sys.stderr)
    results = benchmark.evaluate(model_path=args.model, test_data=args.data)
    print(results.summary())

    if args.output:
        results.save(args.output)
        print(f"\nResults saved to {args.output}", file=sys.stderr)
    return 0


def cmd_info(args) -> int:
    """Handle info command."""
    extras = {
        "torch": "train, eval",
        "transformers": "train, eval",
        "peft": "train",
        "trl": "train",
        "datasets": "train",
        "bitsandbytes": "qlora",
        "spacy": "ner",
    }
    installed = [
        f"  {name:<13} {'installed' if importlib.util.find_spec(name) else 'not installed':<14} ({extra})"
        for name, extra in extras.items()
    ]
    print(
        f"""
Legal LLM Toolkit v{legalkit.__version__}

A Python framework for fine-tuning and evaluating
Large Language Models on legal text.

Features:
  - Citation parsing for UK, US and EU authorities
  - PII anonymisation that preserves legal citations
  - Structure-aware chunking of contracts, judgments and legislation
  - LoRA/QLoRA fine-tuning with legal task presets
  - Legal evaluation benchmarks, including citation grounding

Commands:
  legalkit preprocess  - Clean, anonymise and chunk legal documents
  legalkit anonymise   - Anonymise a document
  legalkit citations   - Extract citations from text
  legalkit train       - Fine-tune a legal LLM
  legalkit evaluate    - Evaluate a legal LLM

Optional dependencies ({install_command("<extra>")}):
{chr(10).join(installed)}

Documentation: {REPOSITORY_URL}
"""
    )
    return 0


def _add_jurisdiction(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j",
        "--jurisdiction",
        default="uk",
        choices=JURISDICTIONS,
        help="Jurisdiction (default: uk)",
    )


def _add_anonymiser_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--salt",
        help="Secret for placeholders that stay the same across documents "
        "(pseudonymisation). Implies --anonymise",
    )
    parser.add_argument("--keep-dates", action="store_true", help="Do not anonymise dates")
    parser.add_argument(
        "--anonymise-case-names",
        action="store_true",
        help="Also anonymise party names inside case citations",
    )
    parser.add_argument(
        "--entities",
        help=f"Comma-separated entity types to anonymise (default: all): {','.join(ENTITY_TYPES)}",
    )
    parser.add_argument("--ner", help="spaCy model for untitled names, e.g. en_core_web_sm")


def _build_anonymiser(args):
    from legalkit.preprocess import Anonymiser

    entity_types = None
    if args.entities:
        entity_types = [e.strip() for e in args.entities.split(",") if e.strip()]
    return Anonymiser(
        preserve_case_names=not args.anonymise_case_names,
        preserve_dates=args.keep_dates,
        salt=args.salt,
        entity_types=entity_types,
        ner=args.ner,
    )


def _read_input(value: str, allow_literal: bool) -> str:
    """Read a file, standard input ("-") or, if allowed, the value itself as text."""
    if value == "-":
        return sys.stdin.read()
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError:  # e.g. text too long to be a file name
        is_file = False
    if is_file:
        return path.read_text(encoding="utf-8")
    if allow_literal:
        return value
    raise FileNotFoundError(f"No such file: {value}")


def _write_output(text: str, output: str | None) -> None:
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _report_entities(entities) -> None:
    counts = Counter(e.entity_type.value for e in entities)
    if counts:
        summary = ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
        print(f"Anonymised {sum(counts.values())} entities: {summary}", file=sys.stderr)
    else:
        print(
            "No personal data detected. Review the text: detection is not exhaustive.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    sys.exit(main())
