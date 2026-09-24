"""
Tests for jurisdiction configurations.
"""

import pytest

from legalkit.jurisdictions import (
    EUJurisdiction,
    UKJurisdiction,
    USJurisdiction,
    get_jurisdiction,
)


@pytest.mark.parametrize(
    "code, cls", [("uk", UKJurisdiction), ("US", USJurisdiction), ("eu", EUJurisdiction)]
)
def test_get_jurisdiction(code, cls):
    config = get_jurisdiction(code)
    assert isinstance(config, cls)
    assert config.code == code.lower()
    assert config.case_citation_patterns
    assert config.legislation_citation_patterns


def test_unknown_jurisdiction():
    with pytest.raises(ValueError, match="Unknown jurisdiction"):
        get_jurisdiction("atlantis")


def test_parse_citation_only_returns_own_jurisdiction():
    text = "[2020] UKSC 1; 347 U.S. 483 (1954); Case C-6/90; section 1 of the Companies Act 2006"
    uk = get_jurisdiction("uk").parse_citation(text)
    assert [(c["type"], c["raw"]) for c in uk] == [
        ("case", "[2020] UKSC 1"),
        ("legislation", "section 1 of the Companies Act 2006"),
    ]
    assert all(c["jurisdiction"] == "uk" for c in uk)
    assert [c["raw"] for c in get_jurisdiction("us").parse_citation(text)] == ["347 U.S. 483"]
    assert [c["raw"] for c in get_jurisdiction("eu").parse_citation(text)] == ["Case C-6/90"]


def test_patterns_are_shared_with_citation_parser():
    from legalkit.preprocess.citations import UK_NEUTRAL

    assert UK_NEUTRAL in get_jurisdiction("uk").case_citation_patterns


@pytest.mark.parametrize(
    "court, level",
    [
        ("UK Supreme Court", 0),
        ("UKSC", 0),
        ("UKHL", 0),
        ("House of Lords", 0),
        ("EWCA Civ", 2),
        ("Court of Appeal (Civil Division)", 2),
        ("EWHC", 3),
        ("Upper Tribunal", 7),
        ("Court of Session", 8),
    ],
)
def test_uk_court_levels(court, level):
    assert get_jurisdiction("uk").get_court_level(court) == level


@pytest.mark.parametrize(
    "court, level",
    [
        ("Supreme Court", 0),
        ("9th Cir.", 1),
        ("S.D.N.Y.", 3),
        ("D. Mass.", 3),
        ("Bankr. S.D.N.Y.", 4),
    ],
)
def test_us_court_levels(court, level):
    assert get_jurisdiction("us").get_court_level(court) == level


def test_eu_court_codes_from_citations():
    eu = get_jurisdiction("eu")
    assert eu.get_court_level("CJ") == 0
    assert eu.get_court_level("GC") == 1


def test_legal_terms():
    uk = get_jurisdiction("uk")
    assert uk.is_legal_term("Claimant")
    assert not uk.is_legal_term("banana")
