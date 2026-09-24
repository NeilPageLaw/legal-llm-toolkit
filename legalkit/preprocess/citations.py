"""
Citation parsing and normalisation for legal documents.

Recognises case citations, legislation references and paragraph pinpoints
in UK, US and EU sources, and records where each one appears in the text.
Every supported format is recognised whatever the parser's primary
jurisdiction, and each Citation is tagged with the jurisdiction of its format.
"""

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class JurisdictionType(Enum):
    """Supported legal jurisdictions."""

    UK = "uk"
    US = "us"
    EU = "eu"
    AU = "au"
    CA = "ca"


SUPPORTED_JURISDICTIONS = ("uk", "us", "eu")

# Citations to the European Convention on Human Rights and the European Human
# Rights Reports are tagged "echr": they are neither EU nor domestic law.
CITATION_JURISDICTIONS = SUPPORTED_JURISDICTIONS + ("echr",)


@dataclass
class Citation:
    """
    A legal citation found in text.

    Attributes:
        raw: The citation exactly as it appears in the text.
        citation_type: "case" or "legislation".
        jurisdiction: Jurisdiction of the citation format ("uk", "us", "eu", "echr").
        parties: Case name, e.g. "Donoghue v Stevenson".
        year: Year of the decision or enactment.
        court: Court code, e.g. "UKSC" or "EWCA Civ".
        volume, reporter, page: Law report details.
        paragraph: Pinpoint paragraph, e.g. "42" or "42-45".
        normalised: Standard form of the citation.
        provision: Cited provision of legislation, e.g. "s 1" or "art 6(1)(f)".
        instrument: Cited legislation, e.g. "Companies Act 2006".
        start, end: Character offsets of ``raw`` in the parsed text.
    """

    raw: str
    citation_type: str
    jurisdiction: str
    parties: str | None = None
    year: int | None = None
    court: str | None = None
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    paragraph: str | None = None
    normalised: str | None = None
    provision: str | None = None
    instrument: str | None = None
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, Any]:
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
            "provision": self.provision,
            "instrument": self.instrument,
            "start": self.start,
            "end": self.end,
        }


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# Neutral citation court codes (England and Wales, UK-wide, Scotland and
# Northern Ireland), in canonical spelling.
UK_COURTS = (
    "UKSC",
    "UKPC",
    "UKHL",
    "EWCA Civ",
    "EWCA Crim",
    "EWHC",
    "EWCOP",
    "EWFC",
    "EWCC",
    "UKUT",
    "UKFTT",
    "UKEAT",
    "EAT",
    "UKIAT",
    "UKAIT",
    "UKSIAC",
    "UKIPTrib",
    "CSIH",
    "CSOH",
    "HCJAC",
    "HCJ",
    "SAC (Civ)",
    "SAC (Crim)",
    "NICA",
    "NIKB",
    "NIQB",
    "NICh",
    "NIFam",
    "NICC",
    "NIMaster",
)

# High Court divisions and tribunal chambers, e.g. "[2024] EWHC 123 (Ch)".
UK_DIVISIONS = (
    "Admin",
    "Admlty",
    "Ch",
    "Comm",
    "Costs",
    "Fam",
    "IPEC",
    "KB",
    "Mercantile",
    "Pat",
    "QB",
    "SCCO",
    "TCC",
    "B",
    "AAC",
    "GRC",
    "IAC",
    "LC",
    "PC",
    "TC",
)

# Law reports cited with the year in square brackets: [1990] 2 AC 605.
UK_REPORTS = (
    "AC",
    "QB",
    "KB",
    "Ch",
    "Fam",
    "P",
    "WLR",
    "All ER",
    "All ER (Comm)",
    "All ER (D)",
    "Lloyd's Rep",
    "Lloyd's Rep IR",
    "Lloyd's Rep PN",
    "Lloyd's Rep Med",
    "BCLC",
    "BCC",
    "Bus LR",
    "Cr App R",
    "Cr App R (S)",
    "Crim LR",
    "ICR",
    "IRLR",
    "EMLR",
    "RPC",
    "FSR",
    "P & CR",
    "HLR",
    "EGLR",
    "BLR",
    "Con LR",
    "CMLR",
    "Costs LR",
    "PNLR",
    "FLR",
    "FCR",
    "STC",
    "Env LR",
    "PIQR",
    "Med LR",
    "BMLR",
    "CLC",
    "WTLR",
    "UKHRR",
    "HRLR",
)

# Reports cited with the year in round brackets: (1854) 9 Exch 341.
UK_ROUND_BRACKET_REPORTS = (
    "App Cas",
    "QBD",
    "Ch D",
    "PD",
    "Ex D",
    "CPD",
    "Exch",
    "HL Cas",
    "Ch App",
    "Cr App R",
    "Cr App R (S)",
    "EHRR",
    "P & CR",
    "HLR",
    "BMLR",
    "LGR",
    "TC",
    "EG",
    "Bing",
    "B & S",
    "E & B",
    "M & W",
    "H & N",
)

REPORT_JURISDICTIONS = {"EHRR": "echr"}

# Law Reports series of 1865-1875: (1868) LR 3 HL 330.
UK_LR_SERIES = ("HL", "QB", "Ch", "CP", "Ex", "Eq", "PC", "Ch App", "A & E", "P & D")

# US reporters in Bluebook form.
US_REPORTS = (
    "U.S.",
    "S. Ct.",
    "L. Ed.",
    "L. Ed. 2d",
    "F.",
    "F.2d",
    "F.3d",
    "F.4th",
    "F. Supp.",
    "F. Supp. 2d",
    "F. Supp. 3d",
    "F. App'x",
    "B.R.",
    "Fed. Cl.",
    "A.2d",
    "A.3d",
    "P.2d",
    "P.3d",
    "N.E.",
    "N.E.2d",
    "N.E.3d",
    "N.W.",
    "N.W.2d",
    "S.E.",
    "S.E.2d",
    "S.W.",
    "S.W.2d",
    "S.W.3d",
    "So.",
    "So. 2d",
    "So. 3d",
    "Cal. Rptr.",
    "Cal. Rptr. 2d",
    "Cal. Rptr. 3d",
    "N.Y.S.",
    "N.Y.S.2d",
    "N.Y.S.3d",
)

EU_COURTS = {"C": "CJ", "T": "GC", "F": "CST"}

# Abbreviations for provisions of UK legislation (OSCOLA style).
PROVISION_ABBREVIATIONS = {
    "section": "s",
    "s": "s",
    "sections": "ss",
    "ss": "ss",
    "regulation": "reg",
    "reg": "reg",
    "regulations": "regs",
    "regs": "regs",
    "rule": "r",
    "r": "r",
    "rules": "rr",
    "rr": "rr",
    "article": "art",
    "art": "art",
    "articles": "arts",
    "arts": "arts",
    "schedule": "sch",
    "sch": "sch",
    "schedules": "schs",
    "schs": "schs",
    "paragraph": "para",
    "para": "para",
    "paragraphs": "paras",
    "paras": "paras",
}


def _squash(value: str) -> str:
    """Comparison key for abbreviations: no spaces, dots or apostrophes, upper case."""
    return re.sub(r"[\s.'’]", "", value).upper()


def _flexible(canonical: str, optional_dots: bool = True) -> str:
    """Regex for an abbreviation that tolerates spacing, full stop and apostrophe variants."""
    parts = []
    for char in canonical:
        if char == " ":
            parts.append(r"\s*")
        elif char == ".":
            parts.append(r"\.?" if optional_dots else r"\.")
        elif char in "'’":
            parts.append(r"['’]?")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def _dotted(canonical: str) -> str:
    """Like _flexible, but also accepts full stops after capitals ("A.C.", "Q.B.D.", "Exch.")."""
    parts = []
    for char in canonical:
        if char == " ":
            parts.append(r"\s*")
        elif char in "'’":
            parts.append(r"['’]?")
        elif char.isupper():
            parts.append(re.escape(char) + r"\.?")
        else:
            parts.append(re.escape(char))
    return "".join(parts) + r"\.?"


def _alternation(canonicals: Iterable[str], transform: Callable[[str], str] = _flexible) -> str:
    """Longest-first alternation so "All ER (Comm)" wins over "All ER"."""
    return "|".join(transform(c) for c in sorted(canonicals, key=len, reverse=True))


def _us_reporter_regex(canonical: str) -> str:
    # Single-letter first-series reporters ("F.") need their full stop,
    # otherwise any "5 F 6" in prose would match.
    single_letter = re.fullmatch(r"[A-Z]\.", canonical) is not None
    return _flexible(canonical, optional_dots=not single_letter)


def _lookup(canonicals: Iterable[str]) -> dict[str, str]:
    return {_squash(c): c for c in canonicals}


_COURT_LOOKUP = _lookup(UK_COURTS)
_DIVISION_LOOKUP = _lookup(UK_DIVISIONS)
_UK_REPORT_LOOKUP = _lookup(UK_REPORTS)
_ROUND_REPORT_LOOKUP = _lookup(UK_ROUND_BRACKET_REPORTS)
_LR_SERIES_LOOKUP = _lookup(UK_LR_SERIES)
_US_REPORT_LOOKUP = _lookup(US_REPORTS)


def _canonical(value: str, lookup: dict[str, str]) -> str:
    return lookup.get(_squash(value), re.sub(r"\s+", " ", value))


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

UK_NEUTRAL = re.compile(
    r"\[(?P<year>\d{4})\]\s+"
    rf"(?P<court>{_alternation(UK_COURTS)})\s+"
    r"(?P<number>B?\d{1,5})\b"
    rf"(?:\s*\((?P<division>{_alternation(UK_DIVISIONS)})\))?",
    re.IGNORECASE,
)

UK_LAW_REPORTS = re.compile(
    r"\[(?P<year>\d{4})\]\s+"
    r"(?:(?P<volume>\d{1,2})\s+)?"
    rf"(?P<reporter>{_alternation(UK_REPORTS, _dotted)})\s+"
    r"(?P<page>\d{1,5})\b",
    re.IGNORECASE,
)

UK_ROUND_BRACKET = re.compile(
    r"\((?P<year>\d{4})\)\s+"
    r"(?P<volume>\d{1,3})\s+"
    rf"(?P<reporter>{_alternation(UK_ROUND_BRACKET_REPORTS, _dotted)})\s+"
    r"(?P<page>\d{1,5})\b"
)

UK_LR_REPORTS = re.compile(
    r"\((?P<year>\d{4})\)\s+LR\s+"
    r"(?P<volume>\d{1,2})\s+"
    rf"(?P<series>{_alternation(UK_LR_SERIES)})\s+"
    r"(?P<page>\d{1,5})\b"
)

# Pinpoint after a UK case citation: "at [42]", ", [42]-[45]", "at para 42".
UK_PARAGRAPH = re.compile(
    r"\s*(?:,\s*)?(?:(?i:at)\s+)?"
    r"(?:\[(?P<p1>\d{1,4})\](?:\s*[-–]\s*\[(?P<p2>\d{1,4})\])?"
    r"|(?i:paras?\.?|paragraphs?)\s*(?P<p3>\d{1,4})(?:\s*[-–]\s*(?P<p4>\d{1,4}))?)"
)

US_CASE = re.compile(
    r"\b(?P<volume>\d{1,4})\s+"
    rf"(?P<reporter>{_alternation(US_REPORTS, _us_reporter_regex)})\s+"
    r"(?P<page>\d{1,5})\b"
)

# Parenthetical after a US citation, optionally after a pinpoint page:
# ", 460 (9th Cir. 2024)" or " (1954)".
US_COURT_YEAR = re.compile(r"(?:,\s*\d+(?:[-–]\d+)?)?\s*\((?P<court>[^()]*?)\s*(?P<year>\d{4})\)")

US_STATUTE = re.compile(
    r"\b(?P<title>\d{1,2})\s+"
    r"(?P<code>U\.?\s?S\.?\s?C\.?(?:\s?A\.?)?|C\.?\s?F\.?\s?R\.?)\s*"
    r"(?:§§?|[Ss]ec(?:tion)?s?\.?|[Pp]arts?)?\s*"
    r"(?P<section>\d+[a-zA-Z]?(?:[.\-–]\d+[a-zA-Z]?)*(?:\([0-9a-zA-Z]{1,4}\))*)"
)

EU_CASE = re.compile(
    r"(?:\b(?i:(?:joined\s+)?cases?)\s+)?"
    r"\b(?P<court>[CTF])[-‑–]\s?(?P<number>\d{1,4})/(?P<year>\d{2})\b"
    r"(?P<appeal>\s+P\b)?"
)

EU_ECLI = re.compile(r"\bECLI:EU:(?P<court>[CTF]):(?P<year>\d{4}):(?P<number>\d+)\b")

EU_ECR = re.compile(r"\[(?P<year>\d{4})\]\s+ECR\s+(?P<page>(?:I{1,3}-)?\d{1,5})\b")

_EU_INSTRUMENT_BODY = (
    r"(?:(?:Council|Commission|Parliament(?:\s+and\s+Council)?)\s+)?"
    r"(?:Implementing\s+|Delegated\s+)?"
    r"(?P<itype>Regulation|Directive|Decision)\s+"
    r"(?:\((?P<prefix>EU|EC|EEC|Euratom)\)\s+)?"
    r"(?P<no>No\.?\s*)?(?P<num1>\d{1,4})/(?P<num2>\d{1,4})"
    r"(?:/(?P<suffix>EU|EC|EEC|Euratom|CFSP|JHA))?"
)

EU_INSTRUMENT = re.compile(rf"\b{_EU_INSTRUMENT_BODY}\b")

_NAMED_INSTRUMENTS = (
    r"UK\s+GDPR|GDPR|TFEU|TEU|ECHR"
    r"|European\s+Convention\s+on\s+Human\s+Rights"
    r"|(?:the\s+)?Charter(?:\s+of\s+Fundamental\s+Rights(?:\s+of\s+the\s+European\s+Union)?)?"
)

_ARTICLE_NUMBER = r"\d+[a-z]?(?:\([0-9a-z]{1,4}\))*"

EU_LEGISLATION = re.compile(
    rf"\b(?i:articles?|arts?)\.?\s*(?P<article>{_ARTICLE_NUMBER})"
    r"(?:,?\s+of\s+(?:the\s+)?|,?\s+)"
    rf"(?P<instrument>(?:{_EU_INSTRUMENT_BODY})|{_NAMED_INSTRUMENTS})\b"
)

# Names of UK legislation: capitalised words, bracketed parts and short
# linking words, ending in Act, Regulations, Rules, Order or Measure.
_ACT_WORD = r"(?:[A-Z][\w'’\-]*|\([^()\n]{1,120}\))"
_ACT_LINK = r"(?:of|and|the|for|to|in|on|with|by|at|from|into|upon|or)"
_ACT_NAME = (
    rf"{_ACT_WORD}(?:,?\s+(?:{_ACT_WORD}|{_ACT_LINK})){{0,15}}?"
    r"\s+(?:Act|Regulations|Rules|Order|Measure)"
)
_PROVISION_NUMBER = r"\d+[A-Z]{0,2}(?:\.\d+)*(?:\([0-9A-Za-z]{1,4}\))*"
_PROVISION_KIND = (
    r"(?i:sections?|ss?|regulations?|regs?|rules?|rr?|articles?|arts?|"
    r"schedules?|schs?|paragraphs?|paras?)"
)

# "section 1 of the Companies Act 2006", "reg 3 of the ... Regulations 2013"
UK_LEGISLATION = re.compile(
    rf"\b(?P<kind>{_PROVISION_KIND})\.?\s*(?P<number>{_PROVISION_NUMBER})"
    rf"(?:\s*(?:,|and|or|to|-|–)\s*{_PROVISION_NUMBER})*"
    r"\s+(?:of|to|in)\s+(?:the\s+)?"
    rf"(?P<name>{_ACT_NAME})(?:\s+(?P<year>\d{{4}}))?\b"
)

# OSCOLA order: "Companies Act 2006, s 994"
UK_LEGISLATION_TRAILING = re.compile(
    rf"\b(?P<name>{_ACT_NAME})\s+(?P<year>\d{{4}}),?\s+"
    rf"(?P<kind>{_PROVISION_KIND})\.?\s*(?P<number>{_PROVISION_NUMBER})"
)

# A named Act or statutory instrument on its own: "the Human Rights Act 1998"
UK_ACT = re.compile(rf"\b(?P<name>{_ACT_NAME})\s+(?P<year>\d{{4}})\b")

# Civil Procedure Rules: "CPR r 3.4", "CPR 31.16", "CPR Part 36", "CPR PD 57AD"
UK_CPR = re.compile(
    r"\bCPR\s+(?:(?P<kind>r|rr|rule|Part|PD)\.?\s*)?"
    r"(?P<number>\d+[A-Z]{0,2}(?:\.\d+)*(?:\([0-9a-z]{1,4}\))*)"
)

# ---------------------------------------------------------------------------
# Case names
# ---------------------------------------------------------------------------

_NAME_WORD = (
    r"(?:(?:Ltd|Co|Inc|Corp|Bros|No|St|Plc|Mr|Mrs|Dr)\."
    r"|[A-Z](?:[\w'’&\-]|\.(?=[A-Za-z]))*"
    r"|\([^()]{1,80}\)"
    r"|&)"
)
# Lower-case words that occur inside party names ("Secretary of State for the
# Home Department", "Marks and Spencer plc"). Prepositions such as "in" are
# deliberately excluded: "the House of Lords in Donoghue v Stevenson".
_NAME_LINKS = (
    "of the and for de la le du des van von der den di da "
    "plc ltd llp inc co on behalf ex parte t/a und"
).split()
_NAME_LINK = "(?:" + "|".join(re.escape(word) for word in _NAME_LINKS) + ")"
_CORPORATE_SUFFIX = r"(?:,\s+(?:Inc|Ltd|LLC|LLP|Co|Corp|N\.A|P\.C|L\.L\.C|L\.P|S\.A|plc)\.?)"
_PARTY = rf"{_NAME_WORD}(?:\s+(?:{_NAME_WORD}|{_NAME_LINK})|{_CORPORATE_SUFFIX})*"

CASE_NAME = re.compile(rf"(?P<p1>{_PARTY})\s+(?:v|vs)\.?\s+(?P<p2>{_PARTY})[\s,]*$")
_IN_RE_NAME = re.compile(rf"\b(?P<prefix>In\s+re|Re|Ex\s+parte)\s+(?P<p1>{_PARTY})[\s,]*$")

# Words that begin a sentence or clause rather than a party name.
_LEADING_STOPWORDS = frozenset(
    """
    a accordingly after also although an and applied applying approved approving as at
    before both but by cf cited citing compare considered considering contrast distinguished
    distinguishing e.g eg either finally followed following for from further furthermore
    he hence her here his however i ie in indeed it its later likewise like moreover
    neither notably of on our overruled overruling per quoted quoting recently see she
    similarly since that the their then there therefore these they this those thus to
    under unlike we when where while yet you
    """.split()
)
_LINK_WORDS = frozenset(_NAME_LINKS)
# Links that cannot end a party name (unlike "plc" or "Ltd").
_DANGLING_LINKS = _LINK_WORDS - {"plc", "ltd", "llp", "inc", "co"}
_ACT_NAME_PREPOSITIONS = frozenset(
    {"of", "for", "to", "from", "by", "in", "on", "with", "at", "into", "upon"}
)


def _clean_party(party: str, first: bool) -> str:
    """Trim a captured party name; only the first party can start mid-sentence."""
    tokens = party.split()
    if first:
        while tokens and (
            tokens[0].lower().rstrip(",") in _LEADING_STOPWORDS or tokens[0].lower() in _LINK_WORDS
        ):
            tokens.pop(0)
    while tokens and tokens[-1].lower() in _DANGLING_LINKS:
        tokens.pop()
    return " ".join(tokens)


def _clean_act_name(name: str) -> str:
    """Trim words captured before the real name of an Act."""
    tokens = re.sub(r"\s+", " ", name).split(" ")
    # "Jones and the Human Rights Act": the name starts after a "the" that is
    # not itself part of the name, as in "Representation of the People Act".
    for index in range(len(tokens) - 1, 0, -1):
        if tokens[index] == "the" and tokens[index - 1].lower() not in _ACT_NAME_PREPOSITIONS:
            tokens = tokens[index + 1 :]
            break
    while tokens and (tokens[0].lower() in _LEADING_STOPWORDS or tokens[0] == "the"):
        tokens.pop(0)
    return " ".join(tokens)


def deduplicate(citations: Iterable[Citation]) -> list[Citation]:
    """Keep the first occurrence of each distinct citation."""
    seen = set()
    unique = []
    for citation in citations:
        key = (citation.citation_type, citation.normalised or citation.raw)
        if key not in seen:
            seen.add(key)
            unique.append(citation)
    return unique


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class CitationParser:
    """
    Parser for legal citations across multiple jurisdictions.

    Supports:
    - UK neutral citations: [2024] UKSC 15, [2023] EWHC 123 (Ch), [2005] UKHL 1
    - UK law reports: [1990] 2 AC 605, [2001] 1 All ER (Comm) 1, (1854) 9 Exch 341
    - UK legislation: section 1 of the Companies Act 2006, Human Rights Act 1998, s 3,
      CPR r 3.4
    - US reporters: 347 U.S. 483 (1954), 123 F.3d 456 (9th Cir. 1997)
    - US statutes: 42 U.S.C. § 1983, 29 C.F.R. § 1630.2
    - EU cases: Case C-311/18, Joined Cases C-6/90 and C-9/90, ECLI:EU:C:2020:559
    - EU legislation: Article 6(1)(f) of Regulation (EU) 2016/679, Article 267 TFEU
    - Paragraph pinpoints: at [42], [42]-[45], para 42

    Example:
        >>> parser = CitationParser(jurisdiction="uk")
        >>> citations = parser.parse("As in Smith v Jones [2024] UKSC 15 at [42]")
        >>> citations[0].court, citations[0].parties, citations[0].paragraph
        ('UKSC', 'Smith v Jones', '42')
    """

    # Public pattern attributes, kept for code that uses them directly.
    UK_NEUTRAL = UK_NEUTRAL
    UK_LAW_REPORTS = UK_LAW_REPORTS
    UK_PARAGRAPH = UK_PARAGRAPH
    CASE_NAME = CASE_NAME
    US_FEDERAL = US_CASE
    US_COURT_YEAR = US_COURT_YEAR
    US_STATUTE = US_STATUTE
    EU_CASE = EU_CASE
    UK_LEGISLATION = UK_LEGISLATION
    EU_LEGISLATION = EU_LEGISLATION

    def __init__(self, jurisdiction: str = "uk", jurisdictions: Iterable[str] | None = None):
        """
        Initialise the citation parser.

        Args:
            jurisdiction: Primary jurisdiction ('uk', 'us' or 'eu').
            jurisdictions: Only report citations tagged with these
                jurisdictions (any of 'uk', 'us', 'eu', 'echr'). Defaults to all.

        Raises:
            ValueError: For an unsupported jurisdiction code.
        """
        self.jurisdiction = jurisdiction.lower()
        if self.jurisdiction not in SUPPORTED_JURISDICTIONS:
            raise ValueError(
                f"Unsupported jurisdiction: {jurisdiction!r}. "
                f"Supported: {', '.join(SUPPORTED_JURISDICTIONS)}"
            )
        self.jurisdictions: frozenset[str] | None = None
        if jurisdictions is not None:
            self.jurisdictions = frozenset(j.lower() for j in jurisdictions)
            unknown = self.jurisdictions - set(CITATION_JURISDICTIONS)
            if unknown:
                raise ValueError(f"Unsupported jurisdictions: {sorted(unknown)}")

    def parse(self, text: str, unique: bool = True) -> list[Citation]:
        """
        Parse all citations from text.

        Args:
            text: Legal text containing citations.
            unique: Return each distinct citation once (its first occurrence).
                With False, every occurrence is returned.

        Returns:
            Citations in order of appearance, with ``start``/``end`` offsets
            into ``text``.
        """
        extractors = (
            self._uk_neutral,
            self._uk_law_reports,
            self._uk_round_bracket,
            self._uk_lr_reports,
            self._uk_legislation,
            self._uk_cpr,
            self._us_cases,
            self._us_statutes,
            self._eu_cases,
            self._eu_ecli,
            self._eu_ecr,
            self._eu_legislation,
            self._eu_instruments,
        )
        found = [citation for extract in extractors for citation in extract(text)]
        if self.jurisdictions is not None:
            found = [c for c in found if c.jurisdiction in self.jurisdictions]

        # Where matches overlap, keep the earliest and then the longest, so
        # "section 1 of the Companies Act 2006" beats "Companies Act 2006".
        found.sort(key=lambda c: (c.start or 0, -(c.end or 0)))
        citations: list[Citation] = []
        for citation in found:
            if citations and (citation.start or 0) < (citations[-1].end or 0):
                continue
            citations.append(citation)

        self._propagate_parallel_parties(text, citations)
        return deduplicate(citations) if unique else citations

    def find_case_names(
        self, text: str, citations: list[Citation] | None = None
    ) -> list[tuple[str, int, int]]:
        """
        Find case names that are immediately followed by a case citation.

        Args:
            text: Text to search.
            citations: Every citation occurrence in ``text``, if already
                parsed with ``parse(text, unique=False)``.

        Returns:
            (name, start, end) tuples with offsets into ``text``.
        """
        if citations is None:
            citations = self.parse(text, unique=False)
        names = []
        for citation in citations:
            if citation.citation_type != "case" or citation.start is None:
                continue
            span = self._case_name_span(text, citation.start)
            if span is not None:
                names.append(span)
        return names

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

    def extract_all(self, text: str) -> dict[str, list[Citation]]:
        """
        Extract and categorise all citations.

        Args:
            text: Legal text to parse

        Returns:
            Dictionary with citations grouped by type
        """
        result: dict[str, list[Citation]] = {"cases": [], "legislation": [], "other": []}
        for c in self.parse(text):
            if c.citation_type == "case":
                result["cases"].append(c)
            elif c.citation_type == "legislation":
                result["legislation"].append(c)
            else:
                result["other"].append(c)
        return result

    # ------------------------------------------------------------------
    # UK
    # ------------------------------------------------------------------

    def _uk_neutral(self, text: str) -> Iterator[Citation]:
        for match in UK_NEUTRAL.finditer(text):
            year = match["year"]
            court = _canonical(match["court"], _COURT_LOOKUP)
            number = match["number"].upper()
            normalised = f"[{year}] {court} {number}"
            if match["division"]:
                normalised += f" ({_canonical(match['division'], _DIVISION_LOOKUP)})"
            yield Citation(
                raw=match.group(0),
                citation_type="case",
                jurisdiction="uk",
                parties=self._find_case_name(text, match.start()),
                year=int(year),
                court=court,
                page=number,
                paragraph=self._find_paragraph(text, match.end()),
                normalised=normalised,
                start=match.start(),
                end=match.end(),
            )

    def _uk_law_reports(self, text: str) -> Iterator[Citation]:
        for match in UK_LAW_REPORTS.finditer(text):
            year, volume, page = match["year"], match["volume"], match["page"]
            reporter = _canonical(match["reporter"], _UK_REPORT_LOOKUP)
            volume_part = f"{volume} " if volume else ""
            yield Citation(
                raw=match.group(0),
                citation_type="case",
                jurisdiction="uk",
                parties=self._find_case_name(text, match.start()),
                year=int(year),
                volume=volume,
                reporter=reporter,
                page=page,
                paragraph=self._find_paragraph(text, match.end()),
                normalised=f"[{year}] {volume_part}{reporter} {page}",
                start=match.start(),
                end=match.end(),
            )

    def _uk_round_bracket(self, text: str) -> Iterator[Citation]:
        for match in UK_ROUND_BRACKET.finditer(text):
            year, volume, page = match["year"], match["volume"], match["page"]
            reporter = _canonical(match["reporter"], _ROUND_REPORT_LOOKUP)
            yield Citation(
                raw=match.group(0),
                citation_type="case",
                jurisdiction=REPORT_JURISDICTIONS.get(reporter, "uk"),
                parties=self._find_case_name(text, match.start()),
                year=int(year),
                volume=volume,
                reporter=reporter,
                page=page,
                paragraph=self._find_paragraph(text, match.end()),
                normalised=f"({year}) {volume} {reporter} {page}",
                start=match.start(),
                end=match.end(),
            )

    def _uk_lr_reports(self, text: str) -> Iterator[Citation]:
        for match in UK_LR_REPORTS.finditer(text):
            year, volume, page = match["year"], match["volume"], match["page"]
            series = _canonical(match["series"], _LR_SERIES_LOOKUP)
            yield Citation(
                raw=match.group(0),
                citation_type="case",
                jurisdiction="uk",
                parties=self._find_case_name(text, match.start()),
                year=int(year),
                volume=volume,
                reporter=f"LR {series}",
                page=page,
                paragraph=self._find_paragraph(text, match.end()),
                normalised=f"({year}) LR {volume} {series} {page}",
                start=match.start(),
                end=match.end(),
            )

    def _uk_legislation(self, text: str) -> Iterator[Citation]:
        for pattern in (UK_LEGISLATION, UK_LEGISLATION_TRAILING):
            for match in pattern.finditer(text):
                name = _clean_act_name(match["name"])
                if not name:
                    continue
                kind = PROVISION_ABBREVIATIONS[match["kind"].lower().rstrip(".")]
                provision = f"{kind} {match['number']}"
                instrument = f"{name} {match['year']}" if match["year"] else name
                name_start = None
                if pattern is UK_LEGISLATION_TRAILING:
                    name_start = match.end("name") - len(name)
                yield self._legislation(
                    match,
                    text,
                    jurisdiction="uk",
                    provision=provision,
                    instrument=instrument,
                    year=match["year"],
                    normalised=f"{instrument}, {provision}",
                    name_start=name_start,
                )

        for match in UK_ACT.finditer(text):
            name = _clean_act_name(match["name"])
            if not name:
                continue
            instrument = f"{name} {match['year']}"
            yield self._legislation(
                match,
                text,
                jurisdiction="uk",
                instrument=instrument,
                year=match["year"],
                normalised=instrument,
                name_start=match.end("name") - len(name),
            )

    def _uk_cpr(self, text: str) -> Iterator[Citation]:
        for match in UK_CPR.finditer(text):
            kind, number = match["kind"], match["number"]
            if kind is None or kind.lower() in ("rule", "r"):
                provision = f"r {number}" if kind or "." in number else f"Part {number}"
            elif kind.lower() == "rr":
                provision = f"rr {number}"
            else:
                provision = f"{kind} {number}"
            yield self._legislation(
                match,
                text,
                jurisdiction="uk",
                provision=provision,
                instrument="CPR",
                normalised=f"CPR {provision}",
            )

    # ------------------------------------------------------------------
    # US
    # ------------------------------------------------------------------

    def _us_cases(self, text: str) -> Iterator[Citation]:
        for match in US_CASE.finditer(text):
            volume, page = match["volume"], match["page"]
            reporter = _canonical(match["reporter"], _US_REPORT_LOOKUP)
            court = year = None
            parenthetical = US_COURT_YEAR.match(text, match.end())
            if parenthetical:
                court = parenthetical["court"].strip(" ,") or None
                year = int(parenthetical["year"])
            yield Citation(
                raw=match.group(0),
                citation_type="case",
                jurisdiction="us",
                parties=self._find_case_name(text, match.start()),
                year=year,
                court=court,
                volume=volume,
                reporter=reporter,
                page=page,
                normalised=f"{volume} {reporter} {page}",
                start=match.start(),
                end=match.end(),
            )

    def _us_statutes(self, text: str) -> Iterator[Citation]:
        for match in US_STATUTE.finditer(text):
            code = _squash(match["code"])
            if code.startswith("CFR"):
                code = "C.F.R."
            else:
                code = "U.S.C.A." if code.endswith("A") else "U.S.C."
            title, section = match["title"], match["section"]
            yield self._legislation(
                match,
                text,
                jurisdiction="us",
                provision=f"§ {section}",
                instrument=f"{title} {code}",
                normalised=f"{title} {code} § {section}",
            )

    # ------------------------------------------------------------------
    # EU
    # ------------------------------------------------------------------

    def _eu_cases(self, text: str) -> Iterator[Citation]:
        for match in EU_CASE.finditer(text):
            court, number, year = match["court"], match["number"], match["year"]
            normalised = f"Case {court}-{number}/{year}" + (" P" if match["appeal"] else "")
            yield Citation(
                raw=match.group(0),
                citation_type="case",
                jurisdiction="eu",
                year=2000 + int(year) if int(year) < 50 else 1900 + int(year),
                court=EU_COURTS[court],
                page=number,
                normalised=normalised,
                start=match.start(),
                end=match.end(),
            )

    def _eu_ecli(self, text: str) -> Iterator[Citation]:
        for match in EU_ECLI.finditer(text):
            court, year, number = match["court"], match["year"], match["number"]
            yield Citation(
                raw=match.group(0),
                citation_type="case",
                jurisdiction="eu",
                year=int(year),
                court=EU_COURTS[court],
                page=number,
                normalised=f"ECLI:EU:{court}:{year}:{number}",
                start=match.start(),
                end=match.end(),
            )

    def _eu_ecr(self, text: str) -> Iterator[Citation]:
        for match in EU_ECR.finditer(text):
            year, page = match["year"], match["page"]
            yield Citation(
                raw=match.group(0),
                citation_type="case",
                jurisdiction="eu",
                parties=self._find_case_name(text, match.start()),
                year=int(year),
                reporter="ECR",
                page=page,
                normalised=f"[{year}] ECR {page}",
                start=match.start(),
                end=match.end(),
            )

    def _eu_legislation(self, text: str) -> Iterator[Citation]:
        for match in EU_LEGISLATION.finditer(text):
            instrument = self._eu_instrument_name(match)
            if instrument == "UK GDPR":
                jurisdiction = "uk"
            elif instrument == "ECHR":
                jurisdiction = "echr"
            else:
                jurisdiction = "eu"
            article = match["article"]
            yield self._legislation(
                match,
                text,
                jurisdiction=jurisdiction,
                provision=f"art {article}",
                instrument=instrument,
                normalised=f"Art {article} {instrument}",
            )

    def _eu_instruments(self, text: str) -> Iterator[Citation]:
        for match in EU_INSTRUMENT.finditer(text):
            instrument = self._eu_instrument_name(match)
            yield self._legislation(
                match, text, jurisdiction="eu", instrument=instrument, normalised=instrument
            )

    @staticmethod
    def _eu_instrument_name(match: re.Match) -> str:
        if match["itype"]:
            prefix = f"({match['prefix']}) " if match["prefix"] else ""
            number = "No " if match["no"] else ""
            suffix = f"/{match['suffix']}" if match["suffix"] else ""
            return f"{match['itype']} {prefix}{number}{match['num1']}/{match['num2']}{suffix}"
        named = re.sub(r"\s+", " ", match["instrument"])
        if named.startswith("the "):
            named = named[4:]
        if named.startswith("Charter"):
            return "Charter"
        if named == "European Convention on Human Rights":
            return "ECHR"
        return named

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _legislation(
        match: re.Match,
        text: str,
        jurisdiction: str,
        normalised: str,
        provision: str | None = None,
        instrument: str | None = None,
        year: str | None = None,
        name_start: int | None = None,
    ) -> Citation:
        start = match.start() if name_start is None else name_start
        return Citation(
            raw=text[start : match.end()],
            citation_type="legislation",
            jurisdiction=jurisdiction,
            year=int(year) if year else None,
            provision=provision,
            instrument=instrument,
            normalised=normalised,
            start=start,
            end=match.end(),
        )

    def _case_name_span(self, text: str, citation_start: int) -> tuple[str, int, int] | None:
        """Find the case name ending just before a citation, with its offsets."""
        window_start = max(0, citation_start - 200)
        window = text[window_start:citation_start]
        match = CASE_NAME.search(window)
        if match is not None:
            p1 = _clean_party(match["p1"], first=True)
            p2 = _clean_party(match["p2"], first=False)
            if not p1 or not p2:
                return None
            name = re.sub(r"\s+", " ", f"{p1} v {p2}")
            start = window_start + match.start("p1") + match["p1"].find(p1.split()[0])
            end = window_start + match.start("p2") + len(match["p2"].rstrip())
            return name, start, end

        match = _IN_RE_NAME.search(window)
        if match is not None:
            p1 = _clean_party(match["p1"], first=False)
            if not p1:
                return None
            name = re.sub(r"\s+", " ", f"{match['prefix']} {p1}")
            end = window_start + match.start("p1") + len(match["p1"].rstrip())
            return name, window_start + match.start("prefix"), end
        return None

    def _find_case_name(self, text: str, citation_start: int) -> str | None:
        """Find case name before a citation position."""
        span = self._case_name_span(text, citation_start)
        return span[0] if span else None

    def _find_paragraph(self, text: str, citation_end: int) -> str | None:
        """Find a paragraph pinpoint immediately after a UK case citation."""
        match = UK_PARAGRAPH.match(text, citation_end)
        if match is None:
            return None
        if match["p1"]:
            # "[2021] AC 100" after a comma is a parallel citation, not a pinpoint.
            bracket = match.start("p1") - 1
            if UK_LAW_REPORTS.match(text, bracket) or UK_NEUTRAL.match(text, bracket):
                return None
            first, last = match["p1"], match["p2"]
        else:
            first, last = match["p3"], match["p4"]
        return f"{first}-{last}" if last else first

    @staticmethod
    def _propagate_parallel_parties(text: str, citations: list[Citation]) -> None:
        """Give parallel citations ("[2020] UKSC 1, [2021] AC 100") the same parties."""
        previous = None
        for citation in citations:
            if citation.citation_type != "case":
                previous = None
                continue
            if (
                citation.parties is None
                and previous is not None
                and previous.parties
                and previous.jurisdiction == citation.jurisdiction
                and re.fullmatch(r"[\s,;]*", text[previous.end : citation.start])
            ):
                citation.parties = previous.parties
            previous = citation
