"""
Citation parsing and normalisation for legal documents.

Handles case citations, legislation references, and paragraph numbers
across UK, US, and EU jurisdictions.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class JurisdictionType(Enum):
    """Supported legal jurisdictions."""
    UK = "uk"
    US = "us"
    EU = "eu"
    AU = "au"
    CA = "ca"


@dataclass
class Citation:
    """Represents a parsed legal citation."""
    raw: str
    citation_type: str  # 'case', 'legislation', 'regulation', 'article'
    jurisdiction: str
    parties: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    paragraph: Optional[str] = None
    normalised: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            "raw": self.raw,
            "type": self.citation_type,
            "jurisdiction": self.jurisdiction,
            "parties": self.parties,
            "year": self.year,
            "court": self.court,
            "volume": self.volume,
            "reporter": self.reporter,
            "page": self.page,
            "paragraph": self.paragraph,
            "normalised": self.normalised,
        }


class CitationParser:
    """
    Parser for legal citations across multiple jurisdictions.
    
    Supports:
    - UK neutral citations: [2024] UKSC 15
    - UK law reports: [2024] 1 AC 123
    - US citations: 123 F.3d 456 (9th Cir. 2024)
    - EU case numbers: Case C-123/24
    - Legislation references: section 1 of the Act 2024
    
    Example:
        >>> parser = CitationParser(jurisdiction="uk")
        >>> citations = parser.parse("As in Smith v Jones [2024] UKSC 15 at [42]")
        >>> print(citations[0].court)
        'UKSC'
    """
    
    # UK neutral citation pattern: [YEAR] COURT NUMBER
    UK_NEUTRAL = re.compile(
        r'\[(\d{4})\]\s+'
        r'(UKSC|UKPC|EWCA\s+(?:Civ|Crim)|EWHC|UKUT|UKFTT|UKEAT|UKIAT)'
        r'\s+(\d+)',
        re.IGNORECASE
    )
    
    # UK law reports: [YEAR] VOLUME REPORTER PAGE
    UK_LAW_REPORTS = re.compile(
        r'\[(\d{4})\]\s+'
        r'(\d+\s+)?'
        r'(AC|QB|Ch|Fam|WLR|All\s*ER|Lloyd\'s\s*Rep)'
        r'\s+(\d+)',
        re.IGNORECASE
    )
    
    # UK paragraph references: at [NUMBER] or para NUMBER or paragraph NUMBER
    UK_PARAGRAPH = re.compile(
        r'(?:at\s+)?\[(\d+)\]|'
        r'(?:para(?:graph)?\.?\s*)(\d+)',
        re.IGNORECASE
    )
    
    # UK case names: Party v Party
    CASE_NAME = re.compile(
        r'([A-Z][a-zA-Z\'\-\s&]+?)\s+v\.?\s+([A-Z][a-zA-Z\'\-\s&]+?)(?=\s*[\[\(]|\s*$)',
        re.MULTILINE
    )
    
    # US Federal Reporter citations: 123 F.3d 456
    US_FEDERAL = re.compile(
        r'(\d+)\s+'
        r'(F\.?\s*(?:2d|3d|4th)?|F\.?\s*Supp\.?\s*(?:2d|3d)?|U\.?S\.?|S\.?\s*Ct\.?)'
        r'\s+(\d+)',
        re.IGNORECASE
    )
    
    # US court and year: (9th Cir. 2024)
    US_COURT_YEAR = re.compile(
        r'\(([^)]*?(?:Cir\.|Ct\.|Dist\.).*?)\s*(\d{4})\)',
        re.IGNORECASE
    )
    
    # EU case numbers: Case C-123/24 or Case T-123/24
    EU_CASE = re.compile(
        r'Case\s+([CT])-(\d+)/(\d{2})',
        re.IGNORECASE
    )
    
    # UK legislation: section X of the Y Act YEAR
    UK_LEGISLATION = re.compile(
        r'(?:section|s\.?)\s*(\d+[A-Z]?(?:\(\d+\))?)'
        r'(?:\s+of\s+(?:the\s+)?)?'
        r'([A-Z][a-zA-Z\s]+(?:Act|Regulations?|Order|Rules?))'
        r'(?:\s+(\d{4}))?',
        re.IGNORECASE
    )
    
    # EU legislation: Article X of Regulation/Directive
    EU_LEGISLATION = re.compile(
        r'(?:Article|Art\.?)\s*(\d+(?:\(\d+\))?)'
        r'(?:\s+of\s+(?:the\s+)?)?'
        r'((?:Regulation|Directive|Decision)\s*(?:\(EU\))?\s*(?:\d+/\d+)?)',
        re.IGNORECASE
    )
    
    def __init__(self, jurisdiction: str = "uk"):
        """
        Initialise the citation parser.
        
        Args:
            jurisdiction: Default jurisdiction for parsing ('uk', 'us', 'eu')
        """
        self.jurisdiction = jurisdiction.lower()
        
    def parse(self, text: str) -> List[Citation]:
        """
        Parse all citations from text.
        
        Args:
            text: Legal text containing citations
            
        Returns:
            List of Citation objects found in text
        """
        citations = []
        
        # Parse based on jurisdiction priority
        if self.jurisdiction == "uk":
            citations.extend(self._parse_uk_citations(text))
            citations.extend(self._parse_uk_legislation(text))
        elif self.jurisdiction == "us":
            citations.extend(self._parse_us_citations(text))
        elif self.jurisdiction == "eu":
            citations.extend(self._parse_eu_citations(text))
            citations.extend(self._parse_eu_legislation(text))
        
        # Always try to parse all jurisdictions for completeness
        if self.jurisdiction != "uk":
            citations.extend(self._parse_uk_citations(text))
        if self.jurisdiction != "us":
            citations.extend(self._parse_us_citations(text))
        if self.jurisdiction != "eu":
            citations.extend(self._parse_eu_citations(text))
            
        # Deduplicate by raw citation text
        seen = set()
        unique = []
        for c in citations:
            if c.raw not in seen:
                seen.add(c.raw)
                unique.append(c)
                
        return unique
    
    def _parse_uk_citations(self, text: str) -> List[Citation]:
        """Parse UK case citations."""
        citations = []
        
        # Find neutral citations
        for match in self.UK_NEUTRAL.finditer(text):
            year, court, number = match.groups()
            raw = match.group(0)
            
            # Look for case name before citation
            parties = self._find_case_name(text, match.start())
            
            # Look for paragraph reference after citation
            para = self._find_paragraph(text, match.end())
            
            citation = Citation(
                raw=raw,
                citation_type="case",
                jurisdiction="uk",
                parties=parties,
                year=int(year),
                court=court.upper().replace(" ", ""),
                paragraph=para,
                normalised=f"[{year}] {court.upper()} {number}"
            )
            citations.append(citation)
        
        # Find law report citations
        for match in self.UK_LAW_REPORTS.finditer(text):
            year, volume, reporter, page = match.groups()
            raw = match.group(0)
            
            parties = self._find_case_name(text, match.start())
            
            citation = Citation(
                raw=raw,
                citation_type="case",
                jurisdiction="uk",
                parties=parties,
                year=int(year),
                volume=volume.strip() if volume else None,
                reporter=reporter.strip(),
                page=page,
                normalised=f"[{year}] {volume or ''}{reporter} {page}".strip()
            )
            citations.append(citation)
            
        return citations
    
    def _parse_uk_legislation(self, text: str) -> List[Citation]:
        """Parse UK legislation references."""
        citations = []
        
        for match in self.UK_LEGISLATION.finditer(text):
            section, act_name, year = match.groups()
            raw = match.group(0)
            
            citation = Citation(
                raw=raw,
                citation_type="legislation",
                jurisdiction="uk",
                year=int(year) if year else None,
                normalised=f"s.{section} {act_name.strip()} {year or ''}".strip()
            )
            citations.append(citation)
            
        return citations
    
    def _parse_us_citations(self, text: str) -> List[Citation]:
        """Parse US case citations."""
        citations = []
        
        for match in self.US_FEDERAL.finditer(text):
            volume, reporter, page = match.groups()
            raw = match.group(0)
            
            # Look for court and year
            court_match = self.US_COURT_YEAR.search(text[match.end():match.end()+50])
            court = court_match.group(1) if court_match else None
            year = int(court_match.group(2)) if court_match else None
            
            citation = Citation(
                raw=raw,
                citation_type="case",
                jurisdiction="us",
                year=year,
                court=court,
                volume=volume,
                reporter=reporter.strip(),
                page=page,
                normalised=f"{volume} {reporter} {page}"
            )
            citations.append(citation)
            
        return citations
    
    def _parse_eu_citations(self, text: str) -> List[Citation]:
        """Parse EU case citations."""
        citations = []
        
        for match in self.EU_CASE.finditer(text):
            court, number, year = match.groups()
            raw = match.group(0)
            
            citation = Citation(
                raw=raw,
                citation_type="case",
                jurisdiction="eu",
                year=2000 + int(year) if int(year) < 50 else 1900 + int(year),
                court=f"ECJ-{court.upper()}",
                normalised=f"Case {court.upper()}-{number}/{year}"
            )
            citations.append(citation)
            
        return citations
    
    def _parse_eu_legislation(self, text: str) -> List[Citation]:
        """Parse EU legislation references."""
        citations = []
        
        for match in self.EU_LEGISLATION.finditer(text):
            article, instrument = match.groups()
            raw = match.group(0)
            
            citation = Citation(
                raw=raw,
                citation_type="legislation",
                jurisdiction="eu",
                normalised=f"Art.{article} {instrument.strip()}"
            )
            citations.append(citation)
            
        return citations
    
    def _find_case_name(self, text: str, citation_pos: int) -> Optional[str]:
        """Find case name before a citation position."""
        # Look at text before citation (up to 100 chars)
        before = text[max(0, citation_pos-100):citation_pos]
        
        match = self.CASE_NAME.search(before)
        if match:
            party1, party2 = match.groups()
            return f"{party1.strip()} v {party2.strip()}"
        return None
    
    def _find_paragraph(self, text: str, citation_pos: int) -> Optional[str]:
        """Find paragraph reference after a citation."""
        # Look at text after citation (up to 20 chars)
        after = text[citation_pos:citation_pos+20]
        
        match = self.UK_PARAGRAPH.search(after)
        if match:
            return match.group(1) or match.group(2)
        return None
    
    def normalise(self, citation: Citation) -> str:
        """
        Normalise a citation to standard format.
        
        Args:
            citation: Citation object to normalise
            
        Returns:
            Normalised citation string
        """
        if citation.normalised:
            return citation.normalised
        return citation.raw
    
    def extract_all(self, text: str) -> Dict[str, List[Citation]]:
        """
        Extract and categorise all citations.
        
        Args:
            text: Legal text to parse
            
        Returns:
            Dictionary with citations grouped by type
        """
        citations = self.parse(text)
        
        result = {
            "cases": [],
            "legislation": [],
            "other": []
        }
        
        for c in citations:
            if c.citation_type == "case":
                result["cases"].append(c)
            elif c.citation_type == "legislation":
                result["legislation"].append(c)
            else:
                result["other"].append(c)
                
        return result
