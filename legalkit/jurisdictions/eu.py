"""
EU jurisdiction configuration.
"""

import re

from legalkit.jurisdictions.base import JurisdictionConfig
from legalkit.preprocess import citations


class EUJurisdiction(JurisdictionConfig):
    """Configuration for EU legal system."""

    def __init__(self):
        super().__init__()

        self.code = "eu"
        self.name = "European Union"

        # Case citations: Case C-123/24, ECLI:EU:C:2024:123, [1991] ECR I-5357
        self.case_citation_patterns = [citations.EU_CASE, citations.EU_ECLI, citations.EU_ECR]

        # Legislation: Article 6(1) of Regulation (EU) 2016/679, Directive 95/46/EC
        self.legislation_citation_patterns = [
            citations.EU_LEGISLATION,
            citations.EU_INSTRUMENT,
        ]

        # EU legal terms
        self.legal_terms = {
            # Parties and actors
            "applicant",
            "defendant",
            "member state",
            "commission",
            "council",
            "parliament",
            "advocate general",
            # Courts and process
            "court of justice",
            "general court",
            "preliminary ruling",
            "preliminary reference",
            "infringement",
            "annulment",
            "opinion",
            "judgment",
            "order",
            # Principles
            "proportionality",
            "subsidiarity",
            "direct effect",
            "indirect effect",
            "supremacy",
            "consistent interpretation",
            "state liability",
            "effectiveness",
            "equivalence",
            # Legislation types
            "regulation",
            "directive",
            "decision",
            "recommendation",
            "treaty",
            "charter",
            # GDPR specific
            "data subject",
            "controller",
            "processor",
            "personal data",
            "processing",
            "consent",
            "legitimate interest",
            "data protection impact assessment",
            "supervisory authority",
            # Competition
            "undertaking",
            "dominant position",
            "abuse",
            "merger",
            "concentration",
            "state aid",
        }

        # EU court hierarchy
        self.court_hierarchy = [
            "Court of Justice",
            "General Court",
        ]

        self.court_aliases = {
            "CJ": "Court of Justice",
            "CJEU": "Court of Justice",
            "ECJ": "Court of Justice",
            "GC": "General Court",
            "CFI": "General Court",
        }

        # Document structure patterns
        self.section_patterns = [
            re.compile(r"^CHAPTER\s+[IVX\d]+", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^SECTION\s+\d+", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^Article\s+\d+", re.MULTILINE | re.IGNORECASE),
        ]
