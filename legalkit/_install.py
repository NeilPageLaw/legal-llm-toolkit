"""
Where the toolkit is installed from.

The toolkit is installed from GitHub and is not published on PyPI. Installing
it by name alone would fetch whatever package someone else had published under
that name, so every install hint includes the GitHub URL.
"""

REPOSITORY_URL = "https://github.com/NeilPageLaw/legal-llm-toolkit"


def install_command(extra: str | None = None) -> str:
    """
    The pip command that installs the toolkit from GitHub.

    Args:
        extra: Optional extra to include, e.g. "train".

    Example:
        >>> print(install_command("train"))
        pip install "legal-llm-toolkit[train] @ git+https://github.com/NeilPageLaw/legal-llm-toolkit"
    """
    requirement = f"legal-llm-toolkit[{extra}]" if extra else "legal-llm-toolkit"
    return f'pip install "{requirement} @ git+{REPOSITORY_URL}"'
