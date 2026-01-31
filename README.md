# Legal LLM Toolkit 🏛️⚖️

A Python framework for fine-tuning and evaluating Large Language Models on legal text.

Built for legal professionals, researchers, and developers who need LLMs that actually understand law.

## Why This Exists

General-purpose fine-tuning tools don't handle legal text well. Legal documents have:

- **Complex citations** — case refs, statute citations, paragraph numbering that varies by jurisdiction
- **Unique structure** — contracts, judgments, and legislation each have their own anatomy
- **Privacy requirements** — training data often needs PII anonymisation (GDPR, professional conduct)
- **Domain-specific evaluation** — "accuracy" means something different for legal reasoning

This toolkit solves these problems.

## Features

- 📚 **Legal Dataset Loaders** — Common legal corpora, easy custom dataset integration
- 🔧 **Legal Preprocessing** — Citation parsing, anonymisation, intelligent chunking
- 🎯 **Fine-tuning Wrappers** — LoRA/QLoRA configs optimised for legal tasks
- 📊 **Legal Evaluation** — Benchmarks for citation accuracy, legal reasoning, jurisdiction-specific tests
- 🌍 **Multi-jurisdiction** — UK, US, EU rule sets out of the box

## Installation

```bash
pip install legal-llm-toolkit
```

Or from source:

```bash
git clone https://github.com/ThePagePage/legal-llm-toolkit.git
cd legal-llm-toolkit
pip install -e .
```

## Quick Start

### Preprocessing Legal Text

```python
from legalkit.preprocess import LegalPreprocessor

processor = LegalPreprocessor(jurisdiction="uk")

# Parse and normalise citations
text = "As held in Smith v Jones [2024] UKSC 15 at [42]..."
processed = processor.process(text)

# Anonymise PII
from legalkit.preprocess import Anonymiser
anon = Anonymiser()
safe_text = anon.anonymise(document)
```

### Fine-tuning a Model

```python
from legalkit.finetune import LegalTrainer, LegalTrainingConfig

config = LegalTrainingConfig(
    base_model="mistralai/Mistral-7B-v0.1",
    method="qlora",
    task="contract_review",
    jurisdiction="uk"
)

trainer = LegalTrainer(config)
trainer.train(dataset="./my_contracts/")
trainer.save("./legal-mistral-contracts")
```

### Evaluating Legal Performance

```python
from legalkit.eval import LegalBenchmark

benchmark = LegalBenchmark(tasks=["citation_accuracy", "legal_reasoning", "contract_qa"])
results = benchmark.evaluate(model_path="./legal-mistral-contracts")
print(results.summary())
```

## Supported Jurisdictions

| Jurisdiction | Citation Parsing | Legislation | Case Law |
|-------------|-----------------|-------------|----------|
| 🇬🇧 UK | ✅ | ✅ | ✅ |
| 🇺🇸 US | ✅ | ✅ | ✅ |
| 🇪🇺 EU | ✅ | ✅ | ✅ |
| 🇦🇺 Australia | 🚧 | 🚧 | 🚧 |
| 🇨🇦 Canada | 🚧 | 🚧 | 🚧 |

## Project Structure

```
legal-llm-toolkit/
├── legalkit/
│   ├── data/           # Dataset loaders, legal corpus handlers
│   ├── preprocess/     # Citation parsing, anonymisation, chunking
│   ├── finetune/       # Training configs, LoRA/QLoRA wrappers
│   ├── eval/           # Legal benchmarks, evaluation metrics
│   └── jurisdictions/  # Jurisdiction-specific rules and patterns
├── examples/           # Sample notebooks and scripts
└── tests/              # Test suite
```

## Contributing

Contributions welcome! Especially:

- Additional jurisdiction support
- Legal benchmark datasets
- Preprocessing improvements
- Documentation and examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this toolkit in research, please cite:

```bibtex
@software{legal_llm_toolkit,
  title = {Legal LLM Toolkit},
  year = {2025},
  url = {https://github.com/ThePagePage/legal-llm-toolkit}
}
```

## Acknowledgements

Built with ❤️ for the legal tech community.

---

**Disclaimer:** This toolkit is for research and development purposes. Always have qualified legal professionals review any outputs used in practice.
