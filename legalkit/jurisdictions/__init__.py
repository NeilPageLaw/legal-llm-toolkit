"""
Jurisdiction-specific rules and patterns.

Provides citation patterns, legal terminology, and
document structure rules for different legal systems.
"""

from legalkit.jurisdictions.base import JurisdictionConfig
from legalkit.jurisdictions.uk import UKJurisdiction
from legalkit.jurisdictions.us import USJurisdiction
from legalkit.jurisdictions.eu import EUJurisdiction


def get_jurisdiction(code: str) -> JurisdictionConfig:
    """
    Get jurisdiction configuration by code.
    
    Args:
        code: Jurisdiction code ('uk', 'us', 'eu')
        
    Returns:
        JurisdictionConfig instance
    """
    jurisdictions = {
        "uk": UKJurisdiction,
        "us": USJurisdiction,
        "eu": EUJurisdiction,
    }
    
    code_lower = code.lower()
    if code_lower not in jurisdictions:
        available = ", ".join(jurisdictions.keys())
        raise ValueError(f"Unknown jurisdiction: {code}. Available: {available}")
        
    return jurisdictions[code_lower]()


__all__ = [
    "JurisdictionConfig",
    "UKJurisdiction",
    "USJurisdiction", 
    "EUJurisdiction",
    "get_jurisdiction",
]
