# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **Package import**: `import legalkit` failed because the `legalkit.data` module was never
  committed (a `.gitignore` entry for `data/` also matched the package).
- **Citations**: case names swallowed the preceding sentence; a following citation's year was
  reported as the pinpoint paragraph; `EWCA Civ` normalised to `EWCA CIV`; UKHL, EWFC, EAT,
  Scottish and NI courts, High Court divisions, KB and pre-1891 reports were not recognised;
  the legislation pattern matched ordinary prose; EU joined cases, ECLI, treaty articles and
  US statutes were not recognised.
- **Anonymisation**: "Mr Smith" and names in capitals ("MR ADAM CARTER") were not anonymised;
  names in party and signature blocks ("Mr Adam Carter\nClaimant") were skipped as legal
  terms; accented names were cut mid-word ("Mr José Álvarez" became "[PERSON_1]é Álvarez");
  names after a no-break space, with particles ("Mr de Souza") or with the surname in capitals
  ("Mr John SMITH") were missed; salutations and offices ("Dear Sir", "Lord Chancellor",
  "Lady Day") were treated as names; organisation and address patterns swallowed whole
  sentences and merged separate companies; overlapping matches corrupted the text; company
  names in preserved case citations were anonymised; the `salt` option was ignored.
- **Chunking**: `chunk_size == overlap` looped forever; clause numbers were deleted and
  paragraphs glued together; chunk offsets were wrong; `min_chunk_size` was ignored.
- **Evaluation**: citation metrics missed `[1990] 2 AC 605`, `EWCA Civ` and `U.S.` citations;
  a response without citations earned a free score; `legal_ner` crashed `summary()`; missing
  test data crashed or scored 0.0; generation was non-deterministic.
- **Fine-tuning**: training failed on current transformers/TRL (`evaluation_strategy`,
  `SFTTrainer(tokenizer=...)`); explicit settings were overwritten by task defaults; a typo in
  `method` started a full fine-tune; saved configs omitted most settings; LoRA models were
  prepared as if quantised; `anonymise_training_data` was skipped for `LegalDataset` inputs;
  full fine-tuning loaded 16-bit weights, so small updates were lost (it now keeps float32
  weights and computes in bf16 or fp16).
- **CLI**: `--chunk-size` was ignored and chunks were never written; long text passed to
  `citations` crashed with "File name too long".

### Added

- `LegalDataset` and `LegalSample` with loaders for directories, JSONL, JSON and Hugging Face
  datasets, and `preprocess`, `chunk`, `deduplicate` and `split(group_by=...)`. Anonymising a
  dataset covers instructions, responses and metadata, with one mapping per sample;
  identifiers named in `keep_metadata` (default `document_id`) are kept. Chunks record a
  `document_id`, so `split(group_by="document_id")` keeps each document in one split; records
  are labelled with their file and line (`cases.jsonl#12`).
- Citation grounding check (`LegalMetrics.evaluate_grounding`) that flags cited authorities
  missing from the source material; ROUGE-L and token F1 metrics; real NER evaluation.
- Citation offsets, provisions and instruments; `CitationParser.find_case_names`; parallel
  citations share their case name; dotted report abbreviations (`[1932] A.C. 562`).
- Anonymisation of postcodes, IBANs and card numbers (checksum-validated), NHS numbers and
  claim numbers; entity selection; optional spaCy NER or custom detectors.
- CLI: `anonymise` command, `train --dry-run/--config/--task/--merge`, `evaluate --data`,
  `--version` and `--verbose`.
- Evaluation test data from a directory of `<task>.jsonl` files, checked for missing fields
  and wrong types before any generation; prompt templates, including the model's own chat
  template, for training and evaluation.
- Saved training configs list their `derived_settings` and the inputs they came from; when a
  config is reused with another method, task or model those settings are derived again,
  unless they were edited.
- GitHub Actions CI (lint, types, tests on Python 3.10-3.13, CPU training tests, build),
  pre-commit hooks, a working quickstart and fictional sample data.

### Changed

- **Installation**: the core install needs only `tqdm`. Training, QLoRA, model evaluation and
  NER are optional extras (`train`, `qlora`, `eval`, `ner`, `all`).
- Python 3.10 or later is required.
- Legislation normalises to OSCOLA style (`Companies Act 2006, s 1`); in-text normalisation
  applies to case citations only. The `court` field uses canonical codes (`EWCA Civ`; `CJ` and
  `GC` for EU courts).
- Unknown jurisdictions, fine-tuning methods and tasks raise `ValueError`. The unimplemented
  `prefix` and `prompt` methods were removed.
- Anonymisation placeholders are numbered in reading order; the mapping holds each original
  form once.
- Chunk overlap defaults to 10% of `chunk_size` (50 for the default of 512) and counts towards
  the chunk size.
- Generation during evaluation is greedy. `format_validity_rate` was removed from citation
  metrics.

### Security

- `trust_remote_code` is off by default for training and evaluation.
- Models pushed to the Hugging Face Hub from the trainer are private by default, and the Hub
  token no longer appears in `repr()` or saved configs.

## [0.1.0] - 2026-01-31

- Initial release.
