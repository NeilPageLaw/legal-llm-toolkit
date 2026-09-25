# Contributing to Legal LLM Toolkit

Thank you for your interest in contributing to Legal LLM Toolkit! This document provides guidelines for contributing.

## Ways to Contribute

- **Bug reports**: Open an issue describing the bug, with the input text and the output you expected
- **Feature requests**: Open an issue describing the feature
- **Code contributions**: Submit a pull request
- **Documentation**: Improve docs, examples, or tutorials
- **Jurisdiction support**: Add support for new legal systems

## Ground Rules for Legal Data

- **Never commit real client or case data**, even anonymised. Tests, examples and issues
  use invented parties and details. `.gitignore` excludes data files outside `examples/`
  and `tests/`, and the pre-commit hooks block large files, but you are responsible for
  what you commit.
- **Every authority you cite must be real and correctly cited.** Check citations used in
  tests, examples, sample data and docs against a primary source such as BAILII,
  The National Archives' Find Case Law or legislation.gov.uk. Invented examples must be
  obviously invented (see `examples/sample_data/`).
- Use phone numbers from Ofcom's drama ranges (e.g. 020 7946 0xxx, 07700 900xxx) and
  addresses such as `example.com` for fictional contact details.

## Development Setup

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/legal-llm-toolkit.git
cd legal-llm-toolkit
```

2. Create a virtual environment (Python 3.10 or later):
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

3. Install in development mode and set up the pre-commit hooks:
```bash
pip install -e ".[dev]"
pre-commit install
```

4. Run tests:
```bash
pytest
```

The end-to-end training and evaluation tests are marked `ml`. They build a tiny model
locally and run on CPU, but need the training extras:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch
pip install -e ".[train,dev]"
pytest -m ml
```

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for formatting and linting, and
  [mypy](https://mypy.readthedocs.io/) for type checking
- Maximum line length is 100 characters
- Use type hints where practical
- Write docstrings for all public functions/classes
- Import heavy libraries (torch, transformers, peft, trl, datasets) inside the functions
  that need them: `import legalkit` must work without them, and a test checks this

Run the same checks as CI:
```bash
ruff check .
ruff format --check .
mypy legalkit
pytest
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Add tests for new functionality, and a regression test for every bug fix
4. Ensure the checks above pass
5. Add a line to the "Unreleased" section of [CHANGELOG.md](CHANGELOG.md)
6. Submit a pull request

## Adding Jurisdiction Support

Citation patterns live in one place, `legalkit/preprocess/citations.py`, and the
jurisdiction configurations reuse them. To add a jurisdiction (e.g. Australia, `au`):

1. In `legalkit/preprocess/citations.py`:
   - Add tables for the jurisdiction's courts and law reports, and the compiled patterns
     built from them
   - Add an extractor method that yields `Citation` objects with `start`/`end` offsets and
     a `normalised` form, and add it to the list in `CitationParser.parse`
   - Add the code to `SUPPORTED_JURISDICTIONS`
2. Create `legalkit/jurisdictions/au.py` with a `JurisdictionConfig` subclass that points
   at those patterns and sets the legal terms, court hierarchy and court aliases
3. Register it in `get_jurisdiction` in `legalkit/jurisdictions/__init__.py`
4. Add tests to `tests/test_citations.py` and `tests/test_jurisdictions.py`, including
   text that must *not* match
5. Update the supported formats table in the README

Example structure:
```python
from legalkit.jurisdictions.base import JurisdictionConfig
from legalkit.preprocess import citations


class AUJurisdiction(JurisdictionConfig):
    def __init__(self):
        super().__init__()
        self.code = "au"
        self.name = "Australia"
        self.case_citation_patterns = [citations.AU_NEUTRAL, citations.AU_LAW_REPORTS]
        self.court_hierarchy = ["High Court", "Full Federal Court", "Federal Court"]
        # ... legislation patterns, legal terms, court aliases
```

## Testing

- Write tests for all new functionality
- Use pytest fixtures for common setup
- Test edge cases and error conditions, and for patterns, false positives
- Assert on exact output: a test that cannot fail does not test anything
- Run the full test suite before submitting PRs

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update examples if API changes (`tests/test_examples.py` runs the quickstart)
- Consider adding a tutorial for complex features

## Reporting Security or Data Protection Issues

If you find a way the toolkit could leak personal data (for example, a kind of personal
data the anonymiser misses), please report it privately to the maintainers rather than in
a public issue, and do not include real personal data in the report.

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing! 🎉
