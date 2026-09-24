"""
Base jurisdiction configuration.
"""

from dataclasses import dataclass, field
from re import Pattern
from typing import Any

from legalkit.preprocess.citations import CitationParser


@dataclass
class JurisdictionConfig:
    """
    Base configuration for a legal jurisdiction.

    Subclasses set jurisdiction-specific citation patterns, terminology,
    court hierarchy and document structure patterns.
    """

    code: str = ""
    name: str = ""

    # Citation patterns (compiled regex, shared with CitationParser)
    case_citation_patterns: list[Pattern] = field(default_factory=list)
    legislation_citation_patterns: list[Pattern] = field(default_factory=list)

    # Legal terminology
    legal_terms: set[str] = field(default_factory=set)

    # Court hierarchy (highest to lowest)
    court_hierarchy: list[str] = field(default_factory=list)

    # Court codes and other names mapped to an entry in court_hierarchy
    court_aliases: dict[str, str] = field(default_factory=dict)

    # Document structure patterns
    section_patterns: list[Pattern] = field(default_factory=list)

    def parse_citation(self, text: str) -> list[dict[str, Any]]:
        """
        Parse citations in this jurisdiction's formats from text.

        Returns:
            One dict per occurrence (see Citation.to_dict), in order of appearance.
        """
        parser = CitationParser(jurisdiction=self.code, jurisdictions=[self.code])
        return [citation.to_dict() for citation in parser.parse(text, unique=False)]

    def is_legal_term(self, word: str) -> bool:
        """Check if word is a recognized legal term."""
        return word.lower() in self.legal_terms

    def get_court_level(self, court_name: str) -> int:
        """
        Get hierarchy level of a court (0 = highest).

        Accepts full names ("Court of Appeal") and codes from citations
        ("EWCA Civ", "UKSC"). Unknown courts rank below every known court.
        """
        court_lower = court_name.strip().lower()
        aliases = {alias.lower(): target for alias, target in self.court_aliases.items()}
        if court_lower in aliases:
            court_lower = aliases[court_lower].lower()
        for i, court in enumerate(self.court_hierarchy):
            if court.lower() in court_lower:
                return i
        return len(self.court_hierarchy)
