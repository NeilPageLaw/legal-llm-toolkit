"""
US jurisdiction configuration.
"""

import re

from legalkit.jurisdictions.base import JurisdictionConfig
from legalkit.preprocess import citations


class USJurisdiction(JurisdictionConfig):
    """Configuration for US legal system."""

    def __init__(self):
        super().__init__()

        self.code = "us"
        self.name = "United States"

        # Reporter citations: 347 U.S. 483, 123 F.3d 456, 5 S. Ct. 6
        self.case_citation_patterns = [citations.US_CASE]

        # Statutes and regulations: 42 U.S.C. § 1983, 29 C.F.R. § 1630.2
        self.legislation_citation_patterns = [citations.US_STATUTE]

        # US legal terms
        self.legal_terms = {
            # Parties
            "plaintiff",
            "defendant",
            "appellant",
            "appellee",
            "petitioner",
            "respondent",
            # Courts and process
            "court",
            "hearing",
            "opinion",
            "order",
            "judgment",
            "injunction",
            "stay",
            "appeal",
            "certiorari",
            "mandamus",
            # Substantive law
            "tort",
            "contract",
            "negligence",
            "breach",
            "duty",
            "causation",
            "damages",
            "liability",
            "indemnification",
            "warranty",
            "covenant",
            "consideration",
            # Constitutional
            "due process",
            "equal protection",
            "commerce clause",
            "first amendment",
            "fourth amendment",
            "fifth amendment",
            # Legal principles
            "precedent",
            "stare decisis",
            "dicta",
            "holding",
            "preemption",
            "standing",
            "mootness",
            "ripeness",
            # Legislation
            "statute",
            "regulation",
            "code",
            "section",
            "subsection",
            # Professionals
            "attorney",
            "counsel",
            "judge",
            "justice",
            "magistrate",
        }

        # US court hierarchy (federal)
        self.court_hierarchy = [
            "Supreme Court",
            "Court of Appeals",
            "Circuit Court",
            "District Court",
            "Bankruptcy Court",
            "Magistrate",
        ]

        self.court_aliases = {"SCOTUS": "Supreme Court"}

        # Document structure patterns
        self.section_patterns = [
            re.compile(r"^ARTICLE\s+[IVX\d]+", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^SECTION\s+\d+", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^§\s*\d+", re.MULTILINE),
        ]

    def get_court_level(self, court_name: str) -> int:
        """
        Get hierarchy level of a court (0 = highest).

        Also understands Bluebook court abbreviations from citation
        parentheticals, e.g. "9th Cir." or "S.D.N.Y.".
        """
        if re.search(r"\bCir\.", court_name):
            return self.court_hierarchy.index("Court of Appeals")
        if re.search(r"\bBankr\.", court_name):
            return self.court_hierarchy.index("Bankruptcy Court")
        if re.fullmatch(r"(?:[NSEWMC]\.)?D\.\s?[A-Z][\w.\s]*", court_name.strip()):
            return self.court_hierarchy.index("District Court")
        return super().get_court_level(court_name)
