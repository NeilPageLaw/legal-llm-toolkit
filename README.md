# Legal LLM Toolkit 🏛️⚖️

[![CI](https://github.com/NeilPageLaw/legal-llm-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/NeilPageLaw/legal-llm-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python framework for preparing legal text, fine-tuning Large Language Models on it,
and evaluating them.

Built for legal professionals, researchers, and developers who need LLMs that actually
understand law.

> **Status: alpha.** The API may change between releases. Review anything this toolkit
> produces before relying on it: see [Data protection and professional
> responsibility](#data-protection-and-professional-responsibility).

## Why This Exists

General-purpose fine-tuning tools don't handle legal text well. Legal documents have:

- **Complex citations**: case references, statute citations and paragraph numbering that
  vary by jurisdiction
- **Unique structure**: contracts, judgments and legislation each have their own anatomy
- **Privacy requirements**: training data often needs personal data removed (UK GDPR,
  professional confidentiality)
- **Domain-specific evaluation**: "accuracy" means something different for legal
  reasoning, and a fabricated citation is worse than no citation

## Features

- 📑 **Citation parsing**: UK neutral citations and law reports (including pre-1891
  reports), UK legislation and the CPR, US reporters and statutes, EU cases, ECLI and
  legislation, treaty and ECHR articles. Case names, courts and pinpoints are extracted,
  every citation gets a normalised form and its exact position in the text.
- 🔒 **Anonymisation**: names, organisations, addresses and postcodes, emails, phone
  numbers, dates, money, bank details (IBAN and card numbers checksum-validated), NI and
  NHS numbers, claim numbers. Citations are never altered. Optional salted pseudonyms
  stable across documents, and an optional spaCy NER model for untitled names.
- ✂️ **Structure-aware chunking**: splits by section, clause and sentence only as far as
  needed, keeps clause numbers, and returns exact character offsets.
- 📚 **Datasets**: load directories, JSONL/JSON and Hugging Face datasets (streaming
  included); clean, anonymise, chunk, deduplicate and split them without leaking one
  document across splits.
- 🎯 **Fine-tuning**: LoRA, QLoRA and full fine-tuning with legal task presets, on current
  and older releases of transformers, TRL and PEFT.
- 📊 **Evaluation**: citation accuracy, a **grounding check that flags cited authorities
  missing from the source material**, legal reasoning, contract QA, summarisation
  (ROUGE-L), legal NER and clause classification.
- 🌍 **Multi-jurisdiction**: UK, US and EU rules out of the box.
- 💻 **Command line**: `legalkit preprocess | anonymise | citations | train | evaluate`.

Preprocessing, citations, anonymisation and metrics are pure Python: the core install
needs no machine-learning libraries.

## Installation

The toolkit is not on PyPI yet; install it from GitHub:

```bash
pip install "legal-llm-toolkit @ git+https://github.com/NeilPageLaw/legal-llm-toolkit"
```

Add extras for the features you need:

| Extra | Adds | Needed for |
|-------|------|------------|
| *(none)* | `tqdm` | Citations, anonymisation, chunking, datasets, metrics, CLI |
| `train` | PyTorch, Transformers, PEFT, TRL, Datasets | Fine-tuning (full and LoRA) |
| `qlora` | `train` + bitsandbytes | QLoRA (needs a CUDA GPU) |
| `eval` | PyTorch, Transformers, PEFT | Evaluating a model with `LegalBenchmark` |
| `ner` | spaCy | Finding untitled names when anonymising |
| `all` | All of the above | |

```bash
pip install "legal-llm-toolkit[train] @ git+https://github.com/NeilPageLaw/legal-llm-toolkit"
```

For development:

```bash
git clone https://github.com/NeilPageLaw/legal-llm-toolkit.git
cd legal-llm-toolkit
pip install -e ".[dev]"
```

## Quick Start

Run the tour of every feature, which needs no machine-learning libraries:

```bash
python examples/quickstart.py
```

### Parsing Citations

```python
from legalkit.preprocess import CitationParser

parser = CitationParser(jurisdiction="uk")
text = "As held in Smith v Jones [2024] UKSC 15 at [42], applying section 1 of the Companies Act 2006"

for citation in parser.parse(text):
    print(citation.normalised, citation.parties, citation.paragraph)
# [2024] UKSC 15 Smith v Jones 42
# Companies Act 2006, s 1 None None
```

### Anonymising Documents

```python
from legalkit.preprocess import Anonymiser

anon = Anonymiser()
result = anon.anonymise("Mr John Smith of 12 High Street, London SW1A 1AA relied on "
                        "Caparo Industries plc v Dickman [1990] 2 AC 605.")
print(result.text)
# [PERSON_1] of [ADDRESS_1], London [ADDRESS_2] relied on Caparo Industries plc v Dickman [1990] 2 AC 605.

original = anon.deanonymise(result.text, result.mapping)  # reverse with the mapping
```

Pass `salt="a secret"` for placeholders that stay the same across documents,
`entity_types=["person", "email"]` to limit what is replaced, and `ner="en_core_web_sm"`
(with the `ner` extra and the spaCy model installed) to catch untitled names.

### Preprocessing and Chunking

```python
from legalkit.preprocess import LegalPreprocessor

processor = LegalPreprocessor(jurisdiction="uk", anonymise=True, chunk_size=512)
result = processor.process(judgment_text, create_chunks=True)

print(len(result.citations), "citations")
print(len(result.chunks), "chunks")
```

### Preparing a Dataset

```python
from legalkit.data import load_legal_corpus

dataset = load_legal_corpus("./my_documents/", jurisdiction="uk")   # or a .jsonl file or HF id
dataset = dataset.deduplicate()
dataset.preprocess(anonymise=True)
chunks = dataset.chunk(chunk_size=512)
train, val, test = chunks.split(train=0.8, val=0.1, test=0.1, group_by="source")
train.to_jsonl("train.jsonl")
```

`group_by="source"` keeps every chunk of a document in the same split, so the test set
never contains text the model was trained on.

### Fine-tuning a Model

```python
from legalkit.finetune import LegalTrainer, LegalTrainingConfig

config = LegalTrainingConfig(
    base_model="mistralai/Mistral-7B-v0.1",
    method="qlora",
    task="contract_review",   # 4096 tokens, 5 epochs unless you set them
    jurisdiction="uk",
)

trainer = LegalTrainer(config)
trainer.train("./my_contracts/")
trainer.save("./legal-mistral-contracts")
```

Instruction data uses `instruction`, `input` and `output` fields (Alpaca layout) and is
formatted with `prompt_template` (`"alpaca"`, `"chatml"`, or `"messages"` to use the
model's own chat template). Check hardware first with
`estimate_memory_usage("mistral-7b", method="qlora")`.

### Evaluating Legal Performance

```python
from legalkit.eval import LegalBenchmark

benchmark = LegalBenchmark(tasks=["citation_accuracy", "contract_qa"], prompt_template="alpaca")
results = benchmark.evaluate(model_path="./legal-mistral-contracts", test_data="./eval_data/")
print(results.summary())
results.save("results.json")
```

Without `test_data`, the benchmark runs a handful of built-in samples: a smoke test, not
a measurement. Any callable `generate_fn(prompt) -> str` can stand in for a model, for
example to evaluate a hosted API.

### Catching Fabricated Citations

```python
from legalkit.eval import LegalMetrics

metrics = LegalMetrics()
check = metrics.evaluate_grounding(model_answer, sources=[source_document])
print(check["ungrounded"])   # authorities cited in the answer but absent from the sources
```

## Command Line

```bash
legalkit citations "As held in Smith v Jones [2024] UKSC 15 at [42]"
legalkit anonymise letter.txt -o letter.anon.txt --mapping mapping.json
legalkit preprocess ./judgments --anonymise --chunk --chunk-size 512 -o train.jsonl
legalkit train train.jsonl -m mistralai/Mistral-7B-v0.1 --method qlora --task legal_qa --dry-run
legalkit evaluate ./legal-llm-output --data ./eval_data --template alpaca -o results.json
legalkit info
```

`legalkit <command> --help` lists every option. `train --dry-run` prints the resolved
configuration and a GPU memory estimate without loading a model.

## Data Formats

**Training data** is a directory of `.txt`/`.md` documents, or JSON Lines records such as:

```json
{"text": "Raw document text for continued pre-training"}
{"instruction": "What is the notice period?", "input": "Either party may terminate ...", "output": "30 days."}
```

`prompt`/`completion` and `question`/`answer` field names are also accepted. Other fields
are kept as metadata.

**Evaluation data** is a directory with one `<task>.jsonl` file per task:

| Task | Fields |
|------|--------|
| `citation_accuracy` | `prompt`, `citations` (list), optional `context` for the grounding check |
| `legal_reasoning` | `question`, `answer`, optional `citations` and `context` |
| `contract_qa` | `context`, `question`, `answers` (list) |
| `case_summarization` | `document`, `summary` |
| `legal_ner` | `text`, `entities` (list of `{"text", "label"}`) |
| `clause_classification` | `text`, `label`, `options` (list) |

See [`examples/sample_data/`](examples/sample_data/) for working files.

## Supported Citation Formats

| Jurisdiction | Cases | Legislation |
|--------------|-------|-------------|
| 🇬🇧 UK | Neutral citations (UKSC, UKPC, UKHL, EWCA Civ/Crim, EWHC with divisions, EWCOP, EWFC, UKUT, UKFTT, EAT, Scottish and NI courts); law reports such as AC, QB, KB, Ch, WLR, All ER, Lloyd's Rep; pre-1891 and LR series | Acts and SIs with sections, regulations, rules, articles and schedules; CPR rules, Parts and PDs; UK GDPR articles |
| 🇺🇸 US | U.S., S. Ct., L. Ed., F., F. Supp., F. App'x and regional reporters, with court and year | U.S.C. and C.F.R. sections |
| 🇪🇺 EU | Case numbers (C-, T-, joined cases), ECLI, ECR | Regulations, directives and decisions; articles of instruments, TFEU, TEU and the Charter |
| ECHR | EHRR reports | Convention articles |
| 🇦🇺 Australia, 🇨🇦 Canada | 🚧 Planned | 🚧 Planned |

## Data Protection and Professional Responsibility

Legal documents often contain confidential and personal data. The toolkit helps, but
cannot take responsibility for how it is used:

- **Anonymisation is not guaranteed.** Rule-based detection finds titled names ("Mr
  Smith"), companies with a legal-form suffix and structured identifiers. Untitled names
  need the optional NER model, and nothing is exhaustive. Review output before sharing it.
- **Salted placeholders are pseudonymisation, not anonymisation.** Under UK GDPR,
  pseudonymised data is still personal data (Recital 26). Keep the salt and any
  `--mapping` file secure: they re-identify the data.
- **Case names in citations are kept by default** because they are public record. If a
  document concerns one of the cited cases, those names identify its parties: use
  `preserve_case_names=False` (`--anonymise-case-names` on the command line).
- **Models can memorise training data.** A model fine-tuned on client material can
  reproduce it. Hub uploads are private by default; do not publish such models. Client
  confidentiality obligations (for solicitors, paragraph 6.3 of the SRA Code of Conduct)
  apply to training data as much as to any other use.
- **Verify every authority a model cites.** `evaluate_grounding` flags citations absent
  from the source material, but a citation that is present may still be misapplied. In
  *R (Ayinde) v London Borough of Haringey* [2025] EWHC 1383 (Admin) the Divisional Court
  warned lawyers about citing fictitious authorities generated by AI.
- **Remote code is off by default.** `trust_remote_code` lets a model repository run code
  on your machine; enable it only for repositories you trust.

## Project Structure

```
legal-llm-toolkit/
├── legalkit/
│   ├── data/           # Datasets, loaders, prompt templates
│   ├── preprocess/     # Citation parsing, anonymisation, chunking, pipeline
│   ├── finetune/       # Training configs, LoRA/QLoRA wrappers, trainer
│   ├── eval/           # Legal benchmarks, evaluation metrics, sample data
│   ├── jurisdictions/  # Jurisdiction-specific rules and court hierarchies
│   └── cli.py          # The legalkit command
├── examples/           # Quickstart script and fictional sample data
└── tests/              # Test suite
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest                  # fast tests, no ML libraries needed
pytest -m ml            # end-to-end training and evaluation (needs the train extra)
ruff check . && ruff format --check . && mypy legalkit
```

The `ml` tests build a tiny model locally, so they need no downloads and run on CPU in
seconds.

## Contributing

Contributions welcome! Especially:

- Additional jurisdiction support (Australia and Canada are next)
- Legal benchmark datasets
- Preprocessing improvements
- Documentation and examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CHANGELOG.md](CHANGELOG.md) for
recent changes.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this toolkit in research, please cite:

```bibtex
@software{legal_llm_toolkit,
  title = {Legal LLM Toolkit},
  author = {{Legal LLM Toolkit Contributors}},
  year = {2025},
  url = {https://github.com/NeilPageLaw/legal-llm-toolkit}
}
```

## Acknowledgements

Built with ❤️ for the legal tech community.

---

**Disclaimer:** This toolkit is for research and development purposes. Always have
qualified legal professionals review any outputs used in practice.
