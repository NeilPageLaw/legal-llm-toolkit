"""
UK jurisdiction configuration.
"""

import re

from legalkit.jurisdictions.base import JurisdictionConfig
from legalkit.preprocess import citations


class UKJurisdiction(JurisdictionConfig):
    """Configuration for UK legal system."""

    def __init__(self):
        super().__init__()

        self.code = "uk"
        self.name = "United Kingdom"

        # Case citations: neutral citations and law reports
        self.case_citation_patterns = [
            citations.UK_NEUTRAL,
            citations.UK_LAW_REPORTS,
            citations.UK_ROUND_BRACKET,
            citations.UK_LR_REPORTS,
        ]

        # Legislation: "section 1 of the Companies Act 2006", "CPR r 3.4"
        self.legislation_citation_patterns = [
            citations.UK_LEGISLATION,
            citations.UK_LEGISLATION_TRAILING,
            citations.UK_ACT,
            citations.UK_CPR,
        ]

        # UK legal terms
        self.legal_terms = {
            # Parties
            "claimant",
            "defendant",
            "appellant",
            "respondent",
            "applicant",
            "petitioner",
            # Courts and process
            "court",
            "tribunal",
            "hearing",
            "judgment",
            "order",
            "injunction",
            "stay",
            "appeal",
            "judicial review",
            # Substantive law
            "tort",
            "contract",
            "negligence",
            "breach",
            "duty of care",
            "causation",
            "remoteness",
            "damages",
            "liability",
            "indemnity",
            "warranty",
            "covenant",
            "consideration",
            # Legal principles
            "precedent",
            "ratio decidendi",
            "obiter dicta",
            "stare decisis",
            "ultra vires",
            "estoppel",
            "laches",
            "equity",
            # Legislation
            "statute",
            "regulation",
            "statutory instrument",
            "act",
            "schedule",
            "section",
            "subsection",
            "paragraph",
            # Professionals
            "solicitor",
            "barrister",
            "counsel",
            "judge",
            "justice",
            "magistrate",
            "tribunal member",
        }

        # UK court hierarchy (England and Wales)
        self.court_hierarchy = [
            "Supreme Court",
            "Privy Council",
            "Court of Appeal",
            "High Court",
            "Crown Court",
            "County Court",
            "Magistrates' Court",
            "Tribunal",
        ]

        self.court_aliases = {
            "UKSC": "Supreme Court",
            "UKHL": "Supreme Court",
            "House of Lords": "Supreme Court",
            "UKPC": "Privy Council",
            "EWCA Civ": "Court of Appeal",
            "EWCA Crim": "Court of Appeal",
            "EWHC": "High Court",
            "EWCOP": "High Court",
            "EWCC": "County Court",
            "UKUT": "Tribunal",
            "UKFTT": "Tribunal",
            "UKEAT": "Tribunal",
            "EAT": "Tribunal",
        }

        # Document structure patterns
        self.section_patterns = [
            re.compile(r"^PART\s+\d+", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^SCHEDULE\s+\d+", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE),
        ]
