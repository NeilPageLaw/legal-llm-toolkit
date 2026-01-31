"""
PII anonymisation for legal documents.

Handles names, addresses, dates, financial information, and other
personally identifiable information while preserving legal structure.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import hashlib


class EntityType(Enum):
    """Types of entities to anonymise."""
    PERSON = "person"
    ORGANISATION = "organisation"
    ADDRESS = "address"
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    MONEY = "money"
    ACCOUNT_NUMBER = "account_number"
    NATIONAL_ID = "national_id"
    CASE_NUMBER = "case_number"


@dataclass
class AnonymisedEntity:
    """Represents an anonymised entity."""
    original: str
    replacement: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class AnonymisationResult:
    """Result of anonymisation process."""
    text: str
    entities: List[AnonymisedEntity] = field(default_factory=list)
    mapping: Dict[str, str] = field(default_factory=dict)
    
    def get_original(self, replacement: str) -> Optional[str]:
        """Get original value from replacement."""
        for orig, repl in self.mapping.items():
            if repl == replacement:
                return orig
        return None


class Anonymiser:
    """
    Anonymises PII in legal documents while preserving legal structure.
    
    Features:
    - Consistent replacement (same name → same placeholder throughout)
    - Preserves legal citations and case names
    - Handles UK/EU data protection requirements
    - Reversible with mapping file
    
    Example:
        >>> anon = Anonymiser()
        >>> result = anon.anonymise("John Smith of 123 High Street signed the contract")
        >>> print(result.text)
        '[PERSON_1] of [ADDRESS_1] signed the contract'
        >>> print(result.mapping)
        {'John Smith': '[PERSON_1]', '123 High Street': '[ADDRESS_1]'}
    """
    
    # Patterns for different entity types
    PATTERNS = {
        EntityType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        EntityType.PHONE: re.compile(
            r'(?:\+44\s?|0)(?:\d\s?){9,10}|'
            r'(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        ),
        EntityType.MONEY: re.compile(
            r'[£$€]\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|m|bn|k))?|'
            r'[\d,]+(?:\.\d{2})?\s*(?:pounds?|dollars?|euros?|GBP|USD|EUR)',
            re.IGNORECASE
        ),
        EntityType.DATE: re.compile(
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|'
            r'July|August|September|October|November|December)\s+\d{4}\b|'
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            re.IGNORECASE
        ),
        EntityType.NATIONAL_ID: re.compile(
            r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b|'  # UK NI number
            r'\b\d{3}-\d{2}-\d{4}\b'  # US SSN
        ),
        EntityType.ACCOUNT_NUMBER: re.compile(
            r'\b\d{8}\b(?=.*sort)|'  # UK account
            r'\b\d{2}-\d{2}-\d{2}\b'  # UK sort code
        ),
        EntityType.ADDRESS: re.compile(
            r'\d+\s+[A-Z][a-zA-Z\s]+(?:Street|Road|Avenue|Lane|Drive|Court|'
            r'Place|Square|Gardens|Crescent|Way|Close|Terrace)\b',
            re.IGNORECASE
        ),
    }
    
    # Common UK/US titles and suffixes for name detection
    TITLES = {'Mr', 'Mrs', 'Ms', 'Miss', 'Dr', 'Prof', 'Sir', 'Dame', 'Lord', 'Lady'}
    
    # Legal terms to preserve (not anonymise)
    LEGAL_PRESERVE = {
        'claimant', 'defendant', 'appellant', 'respondent', 'applicant',
        'plaintiff', 'petitioner', 'court', 'tribunal', 'judge', 'justice',
        'barrister', 'solicitor', 'counsel', 'witness', 'expert'
    }
    
    def __init__(
        self,
        preserve_case_names: bool = True,
        preserve_dates: bool = False,
        consistent_replacement: bool = True,
        salt: Optional[str] = None
    ):
        """
        Initialise the anonymiser.
        
        Args:
            preserve_case_names: Keep case citation names (Smith v Jones)
            preserve_dates: Don't anonymise dates (useful for legal timelines)
            consistent_replacement: Same entity gets same placeholder
            salt: Salt for deterministic hashing (for reproducibility)
        """
        self.preserve_case_names = preserve_case_names
        self.preserve_dates = preserve_dates
        self.consistent_replacement = consistent_replacement
        self.salt = salt or ""
        
        self._counters: Dict[EntityType, int] = {}
        self._mapping: Dict[str, str] = {}
        self._case_names: Set[str] = set()
        
    def reset(self):
        """Reset counters and mappings for new document."""
        self._counters = {t: 0 for t in EntityType}
        self._mapping = {}
        self._case_names = set()
        
    def anonymise(self, text: str, reset: bool = True) -> AnonymisationResult:
        """
        Anonymise PII in text.
        
        Args:
            text: Text to anonymise
            reset: Reset counters for new document
            
        Returns:
            AnonymisationResult with anonymised text and mapping
        """
        if reset:
            self.reset()
            
        entities: List[AnonymisedEntity] = []
        
        # Extract case names first if preserving
        if self.preserve_case_names:
            self._extract_case_names(text)
        
        # Find all entities
        for entity_type, pattern in self.PATTERNS.items():
            if entity_type == EntityType.DATE and self.preserve_dates:
                continue
                
            for match in pattern.finditer(text):
                original = match.group(0)
                
                # Skip if it's a case name we're preserving
                if self._is_case_name(original):
                    continue
                    
                replacement = self._get_replacement(original, entity_type)
                
                entities.append(AnonymisedEntity(
                    original=original,
                    replacement=replacement,
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end()
                ))
        
        # Find person names (more complex pattern)
        entities.extend(self._find_person_names(text))
        
        # Find organisation names
        entities.extend(self._find_organisations(text))
        
        # Sort by position (reverse) for replacement
        entities.sort(key=lambda e: e.start, reverse=True)
        
        # Apply replacements
        result_text = text
        for entity in entities:
            result_text = (
                result_text[:entity.start] + 
                entity.replacement + 
                result_text[entity.end:]
            )
        
        return AnonymisationResult(
            text=result_text,
            entities=sorted(entities, key=lambda e: e.start),
            mapping=dict(self._mapping)
        )
    
    def _get_replacement(self, original: str, entity_type: EntityType) -> str:
        """Get or create replacement for entity."""
        # Normalise for consistent mapping
        key = original.strip().lower()
        
        if self.consistent_replacement and key in self._mapping:
            return self._mapping[key]
        
        self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
        counter = self._counters[entity_type]
        
        replacement = f"[{entity_type.value.upper()}_{counter}]"
        
        if self.consistent_replacement:
            self._mapping[key] = replacement
            self._mapping[original] = replacement
            
        return replacement
    
    def _extract_case_names(self, text: str):
        """Extract case names to preserve."""
        # Pattern: Name v Name (typically in citations)
        case_pattern = re.compile(
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+v\.?\s+'
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)'
            r'(?=\s*[\[\(])'  # Followed by citation
        )
        
        for match in case_pattern.finditer(text):
            self._case_names.add(match.group(1).lower())
            self._case_names.add(match.group(2).lower())
    
    def _is_case_name(self, text: str) -> bool:
        """Check if text is a case name we're preserving."""
        return text.strip().lower() in self._case_names
    
    def _find_person_names(self, text: str) -> List[AnonymisedEntity]:
        """Find person names in text."""
        entities = []
        
        # Pattern: Title + Name or Capitalised Name sequences
        # Mr John Smith, Dr Jane Doe, John William Smith
        name_pattern = re.compile(
            r'\b(?:' + '|'.join(self.TITLES) + r')\.?\s+'
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        )
        
        for match in name_pattern.finditer(text):
            full_match = match.group(0)
            
            # Skip legal terms
            if any(term in full_match.lower() for term in self.LEGAL_PRESERVE):
                continue
                
            # Skip case names
            if self._is_case_name(match.group(1)):
                continue
            
            replacement = self._get_replacement(full_match, EntityType.PERSON)
            
            entities.append(AnonymisedEntity(
                original=full_match,
                replacement=replacement,
                entity_type=EntityType.PERSON,
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
        
        return entities
    
    def _find_organisations(self, text: str) -> List[AnonymisedEntity]:
        """Find organisation names in text."""
        entities = []
        
        # Pattern: Name + Ltd/Limited/PLC/Inc etc
        org_pattern = re.compile(
            r'\b([A-Z][a-zA-Z\s&\-]+?)\s*'
            r'(?:Limited|Ltd|PLC|Inc|LLC|LLP|Corporation|Corp)\b\.?',
            re.IGNORECASE
        )
        
        for match in org_pattern.finditer(text):
            full_match = match.group(0)
            
            replacement = self._get_replacement(full_match, EntityType.ORGANISATION)
            
            entities.append(AnonymisedEntity(
                original=full_match,
                replacement=replacement,
                entity_type=EntityType.ORGANISATION,
                start=match.start(),
                end=match.end(),
                confidence=0.95
            ))
        
        return entities
    
    def deanonymise(self, text: str, mapping: Dict[str, str]) -> str:
        """
        Reverse anonymisation using mapping.
        
        Args:
            text: Anonymised text
            mapping: Original -> Replacement mapping
            
        Returns:
            Original text with PII restored
        """
        result = text
        
        # Create reverse mapping
        reverse = {v: k for k, v in mapping.items() if k == k.strip()}
        
        for replacement, original in reverse.items():
            result = result.replace(replacement, original)
            
        return result
    
    def create_training_pair(
        self, 
        text: str
    ) -> Tuple[AnonymisationResult, Dict[str, str]]:
        """
        Create anonymised training pair with reversible mapping.
        
        Useful for creating training data where you need both
        anonymised and original versions.
        
        Args:
            text: Original text
            
        Returns:
            Tuple of (AnonymisationResult, reverse_mapping)
        """
        result = self.anonymise(text)
        reverse_mapping = {v: k for k, v in result.mapping.items()}
        
        return result, reverse_mapping
