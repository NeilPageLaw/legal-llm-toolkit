# Contributing to Legal LLM Toolkit

Thank you for your interest in contributing to Legal LLM Toolkit! This document provides guidelines for contributing.

## Ways to Contribute

- **Bug reports**: Open an issue describing the bug
- **Feature requests**: Open an issue describing the feature
- **Code contributions**: Submit a pull request
- **Documentation**: Improve docs, examples, or tutorials
- **Jurisdiction support**: Add support for new legal systems

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/legal-llm-toolkit.git
cd legal-llm-toolkit
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest tests/ -v
```

## Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting
- We use [Ruff](https://github.com/astral-sh/ruff) for linting
- Maximum line length is 100 characters
- Use type hints where practical
- Write docstrings for all public functions/classes

Run formatting and linting:
```bash
black legalkit/
ruff check legalkit/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Format your code: `black legalkit/`
7. Submit a pull request

## Adding Jurisdiction Support

To add support for a new jurisdiction:

1. Create a new file in `legalkit/jurisdictions/` (e.g., `au.py` for Australia)
2. Implement the `JurisdictionConfig` class with:
   - Citation patterns (regex)
   - Legal terminology set
   - Court hierarchy
   - Document structure patterns
3. Add the jurisdiction to `legalkit/jurisdictions/__init__.py`
4. Add tests in `tests/test_jurisdictions.py`
5. Update the documentation

Example structure:
```python
from legalkit.jurisdictions.base import JurisdictionConfig

class AUJurisdiction(JurisdictionConfig):
    def __init__(self):
        super().__init__()
        self.code = "au"
        self.name = "Australia"
        # ... implement patterns and terms
```

## Testing

- Write tests for all new functionality
- Use pytest fixtures for common setup
- Test edge cases and error conditions
- Run the full test suite before submitting PRs

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update examples if API changes
- Consider adding a tutorial for complex features

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing! 🎉
