"""
US jurisdiction configuration.
"""

import re
from legalkit.jurisdictions.base import JurisdictionConfig


class USJurisdiction(JurisdictionConfig):
    """Configuration for US legal system."""
    
    def __init__(self):
        super().__init__()
        
        self.code = "us"
        self.name = "United States"
        
        # US citation patterns
        self.case_citation_patterns = [
            # Federal Reporter: 123 F.3d 456
            re.compile(
                r'(\d+)\s+'
                r'(F\.?\s*(?:2d|3d|4th)?|F\.?\s*Supp\.?\s*(?:2d|3d)?)'
                r'\s+(\d+)',
                re.IGNORECASE
            ),
            # US Reports: 123 U.S. 456
            re.compile(
                r'(\d+)\s+U\.?S\.?\s+(\d+)',
                re.IGNORECASE
            ),
            # Supreme Court Reporter: 123 S.Ct. 456
            re.compile(
                r'(\d+)\s+S\.?\s*Ct\.?\s+(\d+)',
                re.IGNORECASE
            ),
        ]
        
        # US legislation patterns
        self.legislation_citation_patterns = [
            # USC: 42 U.S.C. § 1983
            re.compile(
                r'(\d+)\s+U\.?S\.?C\.?\s*§?\s*(\d+)',
                re.IGNORECASE
            ),
            # CFR: 29 C.F.R. § 1630
            re.compile(
                r'(\d+)\s+C\.?F\.?R\.?\s*§?\s*(\d+)',
                re.IGNORECASE
            ),
        ]
        
        # US legal terms
        self.legal_terms = {
            # Parties
            "plaintiff", "defendant", "appellant", "appellee",
            "petitioner", "respondent",
            
            # Courts and process
            "court", "hearing", "opinion", "order", "judgment",
            "injunction", "stay", "appeal", "certiorari", "mandamus",
            
            # Substantive law
            "tort", "contract", "negligence", "breach", "duty",
            "causation", "damages", "liability", "indemnification",
            "warranty", "covenant", "consideration",
            
            # Constitutional
            "due process", "equal protection", "commerce clause",
            "first amendment", "fourth amendment", "fifth amendment",
            
            # Legal principles
            "precedent", "stare decisis", "dicta", "holding",
            "preemption", "standing", "mootness", "ripeness",
            
            # Legislation
            "statute", "regulation", "code", "section", "subsection",
            
            # Professionals
            "attorney", "counsel", "judge", "justice", "magistrate",
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
        
        # Document structure patterns
        self.section_patterns = [
            re.compile(r'^ARTICLE\s+[IVX\d]+', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^SECTION\s+\d+', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^§\s*\d+', re.MULTILINE),
        ]
