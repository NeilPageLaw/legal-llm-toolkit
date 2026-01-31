"""
Base jurisdiction configuration.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Pattern
from dataclasses import dataclass, field


@dataclass
class JurisdictionConfig(ABC):
    """
    Base configuration for a legal jurisdiction.
    
    Subclasses implement jurisdiction-specific patterns
    and rules for legal text processing.
    """
    
    code: str = ""
    name: str = ""
    
    # Citation patterns (compiled regex)
    case_citation_patterns: List[Pattern] = field(default_factory=list)
    legislation_citation_patterns: List[Pattern] = field(default_factory=list)
    
    # Legal terminology
    legal_terms: Set[str] = field(default_factory=set)
    
    # Court hierarchy (highest to lowest)
    court_hierarchy: List[str] = field(default_factory=list)
    
    # Document structure patterns
    section_patterns: List[Pattern] = field(default_factory=list)
    
    def parse_citation(self, text: str) -> List[Dict]:
        """Parse citations from text."""
        citations = []
        
        for pattern in self.case_citation_patterns:
            for match in pattern.finditer(text):
                citations.append({
                    "type": "case",
                    "raw": match.group(0),
                    "jurisdiction": self.code,
                })
                
        for pattern in self.legislation_citation_patterns:
            for match in pattern.finditer(text):
                citations.append({
                    "type": "legislation",
                    "raw": match.group(0),
                    "jurisdiction": self.code,
                })
                
        return citations
    
    def is_legal_term(self, word: str) -> bool:
        """Check if word is a recognized legal term."""
        return word.lower() in self.legal_terms
    
    def get_court_level(self, court_name: str) -> int:
        """Get hierarchy level of a court (0 = highest)."""
        court_lower = court_name.lower()
        for i, court in enumerate(self.court_hierarchy):
            if court.lower() in court_lower:
                return i
        return len(self.court_hierarchy)  # Unknown = lowest
