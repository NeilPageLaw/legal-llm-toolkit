"""
EU jurisdiction configuration.
"""

import re

from legalkit.jurisdictions.base import JurisdictionConfig


class EUJurisdiction(JurisdictionConfig):
    """Configuration for EU legal system."""

    def __init__(self):
        super().__init__()

        self.code = "eu"
        self.name = "European Union"

        # EU citation patterns
        self.case_citation_patterns = [
            # ECJ case numbers: Case C-123/24
            re.compile(r"Case\s+([CT])-(\d+)/(\d{2})", re.IGNORECASE),
            # ECLI: ECLI:EU:C:2024:123
            re.compile(r"ECLI:EU:[CT]:\d{4}:\d+", re.IGNORECASE),
        ]

        # EU legislation patterns
        self.legislation_citation_patterns = [
            # Regulations: Regulation (EU) 2016/679
            re.compile(r"Regulation\s*\((?:EU|EC)\)\s*(?:No\.?\s*)?(\d+)/(\d+)", re.IGNORECASE),
            # Directives: Directive 2019/1024
            re.compile(r"Directive\s*(?:\(EU\)\s*)?(\d+)/(\d+)", re.IGNORECASE),
            # Article references
            re.compile(r"Article\s*(\d+)(?:\((\d+)\))?", re.IGNORECASE),
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

        # Document structure patterns
        self.section_patterns = [
            re.compile(r"^CHAPTER\s+[IVX\d]+", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^SECTION\s+\d+", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^Article\s+\d+", re.MULTILINE | re.IGNORECASE),
        ]
