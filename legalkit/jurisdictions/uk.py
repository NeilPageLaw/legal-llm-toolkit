"""
UK jurisdiction configuration.
"""

import re
from legalkit.jurisdictions.base import JurisdictionConfig


class UKJurisdiction(JurisdictionConfig):
    """Configuration for UK legal system."""
    
    def __init__(self):
        super().__init__()
        
        self.code = "uk"
        self.name = "United Kingdom"
        
        # UK neutral citation patterns
        self.case_citation_patterns = [
            # Neutral citations: [2024] UKSC 15
            re.compile(
                r'\[(\d{4})\]\s+'
                r'(UKSC|UKPC|EWCA\s*(?:Civ|Crim)|EWHC|UKUT|UKFTT|UKEAT|UKIAT)'
                r'\s+(\d+)',
                re.IGNORECASE
            ),
            # Law reports: [2024] 1 AC 123
            re.compile(
                r'\[(\d{4})\]\s+'
                r'(\d+\s+)?'
                r'(AC|QB|Ch|Fam|WLR|All\s*ER|Lloyd\'s\s*Rep)'
                r'\s+(\d+)',
                re.IGNORECASE
            ),
        ]
        
        # UK legislation patterns
        self.legislation_citation_patterns = [
            # Section references: section 1 of the Act 2024
            re.compile(
                r'(?:section|s\.?)\s*(\d+[A-Z]?(?:\(\d+\))?)'
                r'(?:\s+of\s+(?:the\s+)?)?'
                r'([A-Z][a-zA-Z\s]+(?:Act|Regulations?|Order|Rules?))'
                r'(?:\s+(\d{4}))?',
                re.IGNORECASE
            ),
        ]
        
        # UK legal terms
        self.legal_terms = {
            # Parties
            "claimant", "defendant", "appellant", "respondent",
            "applicant", "petitioner",
            
            # Courts and process
            "court", "tribunal", "hearing", "judgment", "order",
            "injunction", "stay", "appeal", "judicial review",
            
            # Substantive law
            "tort", "contract", "negligence", "breach", "duty of care",
            "causation", "remoteness", "damages", "liability",
            "indemnity", "warranty", "covenant", "consideration",
            
            # Legal principles
            "precedent", "ratio decidendi", "obiter dicta", "stare decisis",
            "ultra vires", "estoppel", "laches", "equity",
            
            # Legislation
            "statute", "regulation", "statutory instrument", "act",
            "schedule", "section", "subsection", "paragraph",
            
            # Professionals
            "solicitor", "barrister", "counsel", "judge", "justice",
            "magistrate", "tribunal member",
        }
        
        # UK court hierarchy
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
        
        # Document structure patterns
        self.section_patterns = [
            re.compile(r'^PART\s+\d+', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^SCHEDULE\s+\d+', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\d+\.\s+[A-Z]', re.MULTILINE),
        ]
