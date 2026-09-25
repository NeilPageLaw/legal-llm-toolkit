#!/usr/bin/env python3
"""
Legal LLM Toolkit - Quickstart Example

This script demonstrates the basic usage of the Legal LLM Toolkit
for preprocessing legal documents, extracting citations, and
preparing data for fine-tuning. It needs no machine-learning libraries.
From a copy of the repository:

    pip install .
    python examples/quickstart.py
"""

from pathlib import Path

from legalkit.data import LegalDataset, to_instruction_format
from legalkit.eval import LegalMetrics
from legalkit.finetune import LegalTrainingConfig, estimate_memory_usage
from legalkit.preprocess import Anonymiser, CitationParser, LegalPreprocessor

SAMPLE_DATA = Path(__file__).parent / "sample_data"


def heading(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_citation_parsing():
    """Demonstrate citation parsing."""
    heading("CITATION PARSING")

    parser = CitationParser(jurisdiction="uk")

    text = """
    The principle of duty of care was established in Donoghue v Stevenson
    [1932] AC 562. This was later developed in Caparo Industries plc v
    Dickman [1990] 2 AC 605. More recently, in Robinson v Chief Constable of
    West Yorkshire Police [2018] UKSC 4, the Supreme Court clarified the approach.

    Under section 2(1) of the Unfair Contract Terms Act 1977, liability for
    negligence causing death or personal injury cannot be excluded. See also
    Article 6(1)(f) of Regulation (EU) 2016/679 and 42 U.S.C. § 1983.
    """

    for citation in parser.parse(text):
        print(f"\n  [{citation.citation_type}/{citation.jurisdiction}] {citation.normalised}")
        if citation.parties:
            print(f"    Case:  {citation.parties}")
        if citation.court:
            print(f"    Court: {citation.court}")
        if citation.provision:
            print(f"    Provision: {citation.provision} of {citation.instrument}")


def demo_anonymisation():
    """Demonstrate PII anonymisation."""
    heading("ANONYMISATION")

    contract = (SAMPLE_DATA / "contracts" / "services_agreement.txt").read_text()
    result = Anonymiser().anonymise(contract)

    excerpt = result.text[result.text.index("3.2") : result.text.index("5. LIMITATION")]
    print("\nAnonymised excerpt:\n")
    print(excerpt.strip())

    print("\nEntities found:")
    for original, replacement in result.mapping.items():
        print(f"  {replacement:<20} <- {original}")


def demo_preprocessing():
    """Demonstrate full preprocessing pipeline."""
    heading("PREPROCESSING PIPELINE")

    processor = LegalPreprocessor(jurisdiction="uk", anonymise=True, chunk_size=128)
    judgment = (SAMPLE_DATA / "judgments" / "negligence_appeal.txt").read_text()
    result = processor.process(judgment, create_chunks=True)

    stats = processor.get_statistics(result)
    print(f"\n  Citations found:      {stats['citation_count']}")
    print(f"  Entities anonymised:  {stats['entities_anonymised']}")
    print(f"  Training chunks:      {stats['chunk_count']}")
    print("\n  First chunk:\n")
    print("    " + result.chunks[0][:300].replace("\n", "\n    ") + "...")


def demo_dataset():
    """Demonstrate dataset handling."""
    heading("DATASET HANDLING")

    dataset = LegalDataset.from_directory(SAMPLE_DATA, jurisdiction="uk")
    dataset = dataset.filter(lambda s: "eval" not in (s.source or ""))
    print(f"\n  Loaded {len(dataset)} samples:")
    for key, value in dataset.statistics().items():
        print(f"    {key}: {value}")

    chunked = dataset.chunk(chunk_size=128)
    train, val, test = chunked.split(train=0.8, val=0.1, test=0.1, seed=42, group_by="document_id")
    print(f"\n  After chunking: {len(chunked)} samples")
    print(f"  Split (chunks of one document stay together): {len(train)}/{len(val)}/{len(test)}")

    instruction = next(s for s in dataset if s.is_instruction)
    print("\n  An instruction sample in Alpaca format:\n")
    formatted = to_instruction_format(
        instruction.instruction or "", instruction.response, context=instruction.text
    )
    print("    " + formatted.replace("\n", "\n    "))


def demo_training_config():
    """Demonstrate training configuration."""
    heading("TRAINING CONFIGURATION")

    config = LegalTrainingConfig.for_contract_review(
        base_model="mistralai/Mistral-7B-v0.1",
        method="qlora",
        jurisdiction="uk",
        num_epochs=3,  # overrides the contract_review default of 5
    )
    for key in ("method", "task", "num_epochs", "max_seq_length", "lora_target_modules"):
        print(f"  {key}: {getattr(config, key)}")

    memory = estimate_memory_usage(
        model_name=config.base_model, method=config.method, batch_size=config.batch_size
    )
    print(f"\n  Estimated GPU memory: ~{memory['total_estimated_gb']} GB")
    print(f"  Recommended GPU: {memory['recommended_gpu']}")
    print("\n  To train, on a CUDA GPU with the qlora extra:")
    print(
        '    pip install "legal-llm-toolkit[qlora] @ git+https://github.com/NeilPageLaw/legal-llm-toolkit"'
    )
    print("    LegalTrainer(config).train('examples/sample_data/instructions.jsonl')")


def demo_evaluation():
    """Demonstrate evaluation metrics, including the fabricated-citation check."""
    heading("EVALUATION METRICS")

    metrics = LegalMetrics(jurisdiction="uk")
    source = (SAMPLE_DATA / "judgments" / "negligence_appeal.txt").read_text()

    # A model answer citing one real authority from the source and one
    # authority that appears nowhere in it.
    response = (
        "The duty of care derives from Donoghue v Stevenson [1932] A.C. 562. The claimant "
        "also relies on Carter v Delta Freight [2024] EWCA Civ 9999."
    )
    result = metrics.evaluate_response(
        response=response,
        expected_citations=["[1932] AC 562"],
        sources=source,
    )

    citation = result["citation_metrics"]
    grounding = result["grounding_metrics"]
    print(f"\n  Citation recall:     {citation['recall']:.2f}")
    print(f"  Citation precision:  {citation['precision']:.2f}")
    print(f"  Grounding rate:      {grounding['grounding_rate']:.2f}")
    print(f"  Not in the sources:  {grounding['ungrounded']}  <- check before relying on it")
    print(f"  Aggregate score:     {result['aggregate_score']:.2f}")


def main():
    """Run all demos."""
    demo_citation_parsing()
    demo_anonymisation()
    demo_preprocessing()
    demo_dataset()
    demo_training_config()
    demo_evaluation()
    print("\nDone. See the README for training and evaluation with a real model.\n")


if __name__ == "__main__":
    main()
