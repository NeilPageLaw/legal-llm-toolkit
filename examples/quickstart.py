#!/usr/bin/env python3
"""
Legal LLM Toolkit - Quickstart Example

This script demonstrates the basic usage of the Legal LLM Toolkit
for preprocessing legal documents, extracting citations, and
preparing data for fine-tuning.
"""

from legalkit.eval import LegalMetrics
from legalkit.finetune import LegalTrainingConfig
from legalkit.preprocess import Anonymiser, CitationParser, LegalPreprocessor


def demo_citation_parsing():
    """Demonstrate citation parsing."""
    print("=" * 60)
    print("CITATION PARSING DEMO")
    print("=" * 60)

    parser = CitationParser(jurisdiction="uk")

    text = """
    The principle of duty of care was established in Donoghue v Stevenson 
    [1932] AC 562. This was later developed in Caparo Industries plc v 
    Dickman [1990] 2 AC 605, where the House of Lords set out the 
    three-stage test. More recently, in Robinson v Chief Constable of 
    West Yorkshire [2018] UKSC 4, the Supreme Court clarified the approach.
    
    Under section 2 of the Unfair Contract Terms Act 1977, liability for 
    negligence cannot be excluded in certain circumstances.
    """

    citations = parser.parse(text)

    print(f"\nFound {len(citations)} citations:\n")
    for citation in citations:
        print(f"  Type: {citation.citation_type}")
        print(f"  Raw: {citation.raw}")
        if citation.normalised:
            print(f"  Normalised: {citation.normalised}")
        if citation.year:
            print(f"  Year: {citation.year}")
        if citation.court:
            print(f"  Court: {citation.court}")
        print()


def demo_anonymisation():
    """Demonstrate PII anonymisation."""
    print("=" * 60)
    print("ANONYMISATION DEMO")
    print("=" * 60)

    anonymiser = Anonymiser(preserve_case_names=True, preserve_dates=False)

    text = """
    CONFIDENTIAL
    
    Re: Smith v Jones [2024] UKSC 15
    
    Dear Mr John Williams,
    
    Further to our telephone conversation on 15 January 2024, I write to 
    confirm that ABC Limited has instructed us to act in this matter.
    
    Please send all correspondence to john.williams@lawfirm.co.uk or 
    contact us on 020 7123 4567.
    
    The claim is for £150,000 in damages.
    
    Yours sincerely,
    Jane Smith
    Partner
    """

    result = anonymiser.anonymise(text)

    print("\nOriginal text:")
    print("-" * 40)
    print(text[:300] + "...")

    print("\nAnonymised text:")
    print("-" * 40)
    print(result.text[:300] + "...")

    print("\nEntity mapping:")
    print("-" * 40)
    for original, replacement in list(result.mapping.items())[:5]:
        print(f"  {original} → {replacement}")


def demo_preprocessing():
    """Demonstrate full preprocessing pipeline."""
    print("=" * 60)
    print("PREPROCESSING PIPELINE DEMO")
    print("=" * 60)

    processor = LegalPreprocessor(
        jurisdiction="uk", anonymise=True, normalise_citations=True, normalise_whitespace=True
    )

    document = """
    JUDGEMENT
    
    Smith v Jones [2024] UKSC 15
    
    LORD SMITH (with whom Lord Jones agrees):
    
    1. This appeal concerns the interpretation of section 1 of the 
    Contracts Act 2020. The appellant, Mr John Davies of 123 High Street, 
    London, contends that the lower courts erred in their application 
    of Donoghue v Stevenson [1932] AC 562.
    
    2. For the reasons I shall give, I would dismiss this appeal.
    
    THE FACTS
    
    3. The relevant facts are as follows. On 1 January 2023, the 
    respondent, ABC Limited, entered into a contract with the appellant...
    """

    result = processor.process(document, create_chunks=True)

    print("\nProcessing results:")
    print(f"  Original length: {len(result.original)} chars")
    print(f"  Processed length: {len(result.processed)} chars")
    print(f"  Citations found: {len(result.citations)}")
    print(f"  Chunks created: {len(result.chunks)}")

    if result.anonymisation:
        print(f"  Entities anonymised: {len(result.anonymisation.entities)}")

    print("\nCitations extracted:")
    for c in result.citations[:3]:
        print(f"  - {c.normalised or c.raw} ({c.citation_type})")


def demo_dataset():
    """Demonstrate dataset handling."""
    print("=" * 60)
    print("DATASET HANDLING DEMO")
    print("=" * 60)

    from legalkit.data.dataset import LegalDataset, LegalSample

    # Create sample dataset
    samples = [
        LegalSample(
            text="This agreement is between Party A and Party B...",
            document_type="contract",
            jurisdiction="uk",
        ),
        LegalSample(
            text="The court held that the defendant was liable...",
            document_type="case",
            jurisdiction="uk",
        ),
        LegalSample(
            text="Section 1 provides that all persons shall...",
            document_type="legislation",
            jurisdiction="uk",
        ),
    ]

    dataset = LegalDataset(samples)

    print("\nDataset statistics:")
    stats = dataset.statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Split dataset
    train, val, test = dataset.split(train=0.6, val=0.2, test=0.2, seed=42)
    print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}")


def demo_training_config():
    """Demonstrate training configuration."""
    print("=" * 60)
    print("TRAINING CONFIGURATION DEMO")
    print("=" * 60)

    # Create config for contract review task
    config = LegalTrainingConfig.for_contract_review(
        base_model="mistralai/Mistral-7B-v0.1",
        method="qlora",
        jurisdiction="uk",
        learning_rate=2e-4,
        num_epochs=3,
    )

    print("\nTraining configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Estimate memory usage
    from legalkit.finetune.adapters import estimate_memory_usage

    memory = estimate_memory_usage(model_name="mistral-7b", method="qlora", batch_size=4)

    print("\nEstimated memory usage:")
    for key, value in memory.items():
        print(f"  {key}: {value}")


def demo_evaluation():
    """Demonstrate evaluation metrics."""
    print("=" * 60)
    print("EVALUATION METRICS DEMO")
    print("=" * 60)

    metrics = LegalMetrics(jurisdiction="uk")

    # Evaluate a model response
    response = """
    The test for negligence was established in Donoghue v Stevenson [1932] AC 562,
    which requires proof of: (1) a duty of care owed by the defendant to the 
    claimant; (2) breach of that duty; (3) causation; and (4) damage.
    
    This was further developed in Caparo Industries plc v Dickman [1990] 2 AC 605,
    where the House of Lords established a three-stage test: foreseeability,
    proximity, and whether it is fair, just and reasonable to impose a duty.
    """

    reference = """
    The elements of negligence are: duty of care, breach, causation, and damage.
    Key cases include Donoghue v Stevenson and Caparo v Dickman.
    """

    result = metrics.evaluate_response(
        response=response,
        reference=reference,
        expected_citations=["[1932] AC 562", "[1990] 2 AC 605"],
    )

    print("\nEvaluation results:")
    print(f"  Aggregate score: {result['aggregate_score']:.3f}")

    print("\nCitation metrics:")
    for key, value in result["citation_metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\nTerminology metrics:")
    print(f"  Legal terms found: {result['terminology_metrics']['legal_term_count']}")
    print(f"  Terms: {', '.join(result['terminology_metrics']['terms_used'][:5])}...")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("LEGAL LLM TOOLKIT - QUICKSTART DEMO")
    print("=" * 60 + "\n")

    demo_citation_parsing()
    print("\n")

    demo_anonymisation()
    print("\n")

    demo_preprocessing()
    print("\n")

    demo_dataset()
    print("\n")

    demo_training_config()
    print("\n")

    demo_evaluation()

    print("\n" + "=" * 60)
    print("Demo complete! See the documentation for more details.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
