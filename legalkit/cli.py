"""
Legal LLM Toolkit CLI.

Command-line interface for common toolkit operations.
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="legalkit",
        description="Legal LLM Toolkit - Fine-tune and evaluate LLMs on legal text"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess legal documents"
    )
    preprocess_parser.add_argument("input", help="Input file or directory")
    preprocess_parser.add_argument("-o", "--output", help="Output file")
    preprocess_parser.add_argument(
        "-j", "--jurisdiction",
        default="uk",
        help="Jurisdiction (uk, us, eu)"
    )
    preprocess_parser.add_argument(
        "--anonymise",
        action="store_true",
        help="Anonymise PII"
    )
    preprocess_parser.add_argument(
        "--chunk",
        action="store_true",
        help="Create training chunks"
    )
    preprocess_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Fine-tune a legal LLM"
    )
    train_parser.add_argument("dataset", help="Training dataset path")
    train_parser.add_argument(
        "-m", "--model",
        default="mistralai/Mistral-7B-v0.1",
        help="Base model"
    )
    train_parser.add_argument(
        "--method",
        default="qlora",
        choices=["full", "lora", "qlora"],
        help="Fine-tuning method"
    )
    train_parser.add_argument(
        "-o", "--output",
        default="./legal-llm-output",
        help="Output directory"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    train_parser.add_argument(
        "-j", "--jurisdiction",
        default="uk",
        help="Jurisdiction"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a legal LLM"
    )
    eval_parser.add_argument("model", help="Model path")
    eval_parser.add_argument(
        "--tasks",
        nargs="+",
        help="Evaluation tasks"
    )
    eval_parser.add_argument(
        "-o", "--output",
        help="Output file for results"
    )
    eval_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per task"
    )
    
    # Parse citations command
    cite_parser = subparsers.add_parser(
        "citations",
        help="Extract citations from text"
    )
    cite_parser.add_argument("input", help="Input file or text")
    cite_parser.add_argument(
        "-j", "--jurisdiction",
        default="uk",
        help="Jurisdiction"
    )
    cite_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show toolkit information"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    try:
        if args.command == "preprocess":
            return cmd_preprocess(args)
        elif args.command == "train":
            return cmd_train(args)
        elif args.command == "evaluate":
            return cmd_evaluate(args)
        elif args.command == "citations":
            return cmd_citations(args)
        elif args.command == "info":
            return cmd_info(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_preprocess(args):
    """Handle preprocess command."""
    from legalkit.preprocess import LegalPreprocessor
    from legalkit.data import LegalDataset
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        dataset = LegalDataset.from_directory(input_path)
    elif input_path.suffix == ".jsonl":
        dataset = LegalDataset.from_jsonl(input_path)
    else:
        # Single file
        text = input_path.read_text()
        processor = LegalPreprocessor(
            jurisdiction=args.jurisdiction,
            anonymise=args.anonymise
        )
        result = processor.process(text, create_chunks=args.chunk)
        
        if args.output:
            Path(args.output).write_text(result.processed)
        else:
            print(result.processed)
            
        if result.citations:
            print(f"\nFound {len(result.citations)} citations:", file=sys.stderr)
            for c in result.citations:
                print(f"  - {c.normalised or c.raw}", file=sys.stderr)
                
        return 0
    
    # Process dataset
    dataset.preprocess(
        anonymise=args.anonymise,
        jurisdiction=args.jurisdiction
    )
    
    if args.output:
        dataset.to_jsonl(args.output)
        print(f"Saved {len(dataset)} documents to {args.output}")
    else:
        stats = dataset.statistics()
        print(json.dumps(stats, indent=2))
        
    return 0


def cmd_train(args):
    """Handle train command."""
    from legalkit.finetune import LegalTrainer, LegalTrainingConfig
    
    config = LegalTrainingConfig(
        base_model=args.model,
        method=args.method,
        jurisdiction=args.jurisdiction,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output,
    )
    
    print(f"Training configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    print()
    
    trainer = LegalTrainer(config)
    
    print(f"Starting training on {args.dataset}...")
    result = trainer.train(args.dataset)
    
    print(f"\nTraining complete!")
    print(f"Final loss: {result['train_loss']:.4f}")
    
    trainer.save(args.output)
    print(f"Model saved to {args.output}")
    
    return 0


def cmd_evaluate(args):
    """Handle evaluate command."""
    from legalkit.eval import LegalBenchmark
    
    benchmark = LegalBenchmark(
        tasks=args.tasks,
        max_samples=args.max_samples,
    )
    
    print(f"Running evaluation on {args.model}...")
    results = benchmark.evaluate(model_path=args.model)
    
    print(results.summary())
    
    if args.output:
        results.save(args.output)
        print(f"\nResults saved to {args.output}")
        
    return 0


def cmd_citations(args):
    """Handle citations command."""
    from legalkit.preprocess import CitationParser
    
    input_path = Path(args.input)
    
    if input_path.exists():
        text = input_path.read_text()
    else:
        text = args.input
    
    parser = CitationParser(jurisdiction=args.jurisdiction)
    citations = parser.parse(text)
    
    if args.json:
        print(json.dumps([c.to_dict() for c in citations], indent=2))
    else:
        if citations:
            print(f"Found {len(citations)} citations:\n")
            for c in citations:
                print(f"  [{c.citation_type}] {c.normalised or c.raw}")
                if c.parties:
                    print(f"           Case: {c.parties}")
                if c.year:
                    print(f"           Year: {c.year}")
                print()
        else:
            print("No citations found.")
            
    return 0


def cmd_info(args):
    """Handle info command."""
    import legalkit
    
    print(f"""
Legal LLM Toolkit v{legalkit.__version__}

A Python framework for fine-tuning and evaluating
Large Language Models on legal text.

Features:
  - Legal text preprocessing (citations, anonymisation, chunking)
  - Dataset handling for contracts, cases, and legislation
  - LoRA/QLoRA fine-tuning with legal-optimized configs
  - Legal-specific evaluation benchmarks
  - Multi-jurisdiction support (UK, US, EU)

Commands:
  legalkit preprocess  - Preprocess legal documents
  legalkit train       - Fine-tune a legal LLM
  legalkit evaluate    - Evaluate a legal LLM
  legalkit citations   - Extract citations from text

Documentation: https://github.com/YOUR_USERNAME/legal-llm-toolkit
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
