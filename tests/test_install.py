"""
The toolkit is not published on PyPI, so every install hint must use GitHub.
"""

from pathlib import Path

from legalkit._install import REPOSITORY_URL, install_command
from legalkit.finetune.trainer import INSTALL_HINT

ROOT = Path(__file__).parent.parent


def test_install_command_uses_github():
    assert install_command() == f'pip install "legal-llm-toolkit @ git+{REPOSITORY_URL}"'
    assert install_command("train") == (
        f'pip install "legal-llm-toolkit[train] @ git+{REPOSITORY_URL}"'
    )
    assert install_command("train") in INSTALL_HINT


def test_no_install_hint_points_at_pypi():
    """A hint without the GitHub URL would install another publisher's package."""
    files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    files += sorted((ROOT / "legalkit").rglob("*.py")) + sorted((ROOT / "examples").rglob("*.py"))
    hints = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "pip install" in line and "legal-llm-toolkit" in line and "@ git+" not in line
    ]
    assert hints == []
