"""
PII anonymisation for legal documents.

Handles names, addresses, dates, financial information, and other
personally identifiable information while preserving legal structure.
"""

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum

from legalkit.preprocess.citations import CitationParser


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
    entities: list[AnonymisedEntity] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)

    def get_original(self, replacement: str) -> str | None:
        """Get original value from replacement."""
        for orig, repl in self.mapping.items():
            if repl == replacement:
                return orig
        return None


# A detector finds entities the rules cannot, e.g. untitled names. It returns
# (start, end, label) spans, where label is an EntityType value or an NER
# label such as "PERSON" or "ORG".
Detector = Callable[[str], Iterable[tuple[int, int, str]]]

NER_LABELS = {
    "PERSON": EntityType.PERSON,
    "PER": EntityType.PERSON,
    "ORG": EntityType.ORGANISATION,
    "ORGANIZATION": EntityType.ORGANISATION,
    "ORGANISATION": EntityType.ORGANISATION,
    "EMAIL_ADDRESS": EntityType.EMAIL,
    "PHONE_NUMBER": EntityType.PHONE,
    "IBAN_CODE": EntityType.ACCOUNT_NUMBER,
    "CREDIT_CARD": EntityType.ACCOUNT_NUMBER,
    "UK_NHS": EntityType.NATIONAL_ID,
    "US_SSN": EntityType.NATIONAL_ID,
}


def _iban_is_valid(value: str) -> bool:
    """Check an IBAN's mod-97 check digits."""
    compact = re.sub(r"\s", "", value)
    if not compact.isalnum():
        return False
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(int(char, 36)) for char in rearranged)
    return int(digits) % 97 == 1


def _luhn_is_valid(value: str) -> bool:
    """Check a payment card number with the Luhn algorithm."""
    digits = [int(d) for d in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


@dataclass(frozen=True)
class _Rule:
    """A detection rule. A pattern group named "value" narrows the entity span."""

    entity_type: EntityType
    pattern: re.Pattern
    validator: Callable[[str], bool] | None = None


_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
# "the 2019 High Court judgment" names a court, not an address.
_COURT_TYPES = (
    r"(?:High|Crown|County|Supreme|Magistrates['’]?|Family|Divisional|Commercial|Admiralty|"
    r"Chancery|Coroners?['’]?|Youth|Circuit|District|Appeals?|Tax|Employment|Upper|Business|"
    r"Property|Patents)"
)
_STREET_TYPES = (
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Square|Sq|"
    r"Gardens|Gdns|Crescent|Cres|Way|Close|Terrace|Grove|Hill|Park|Row|Mews|Walk|"
    r"Parade|Rise|View|Green|Boulevard|Blvd|Wharf|Quay|Yard|Circus)"
)

# Rules in priority order: when two matches cover the same text, the earlier
# rule wins (a sort code beats a date that looks the same).
RULES = (
    _Rule(
        EntityType.EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    _Rule(
        EntityType.NATIONAL_ID,
        # UK National Insurance number
        re.compile(
            r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]"
            r"\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"
        ),
    ),
    _Rule(EntityType.NATIONAL_ID, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),  # US SSN
    _Rule(
        EntityType.NATIONAL_ID,
        re.compile(r"(?i:\bNHS\s*(?:no\.?|number)?\s*:?\s*)(?P<value>\d{3}\s?\d{3}\s?\d{4})\b"),
    ),
    _Rule(EntityType.ACCOUNT_NUMBER, re.compile(r"\b\d{2}-\d{2}-\d{2}\b")),  # UK sort code
    _Rule(
        EntityType.ACCOUNT_NUMBER,
        re.compile(r"(?i:\baccount\s*(?:no\.?|number)?\s*:?\s*)(?P<value>\d{8})\b"),
    ),
    _Rule(
        EntityType.ACCOUNT_NUMBER,
        re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?\b"),
        validator=_iban_is_valid,
    ),
    _Rule(
        EntityType.ACCOUNT_NUMBER,
        re.compile(r"\b(?:\d{4}[ -]?){3}\d{1,7}\b"),
        validator=_luhn_is_valid,
    ),
    _Rule(
        EntityType.CASE_NUMBER,
        # Court claim numbers: HC-2014-000123, CO/1234/2020, C1/2020/1234
        re.compile(r"\b(?:[A-Z]{2}-\d{4}-\d{6}|[A-Z]{1,3}\d?/\d{4}/\d{2,5})\b"),
    ),
    _Rule(
        EntityType.PHONE,
        # UK: 020 7123 4567, 07123 456789, +44 (0)20 7123 4567
        re.compile(r"(?<![\d+])(?:\+44\s?(?:\(0\)\s?)?|0)(?:\d[\s-]?){8,9}\d(?!\d)"),
    ),
    _Rule(
        EntityType.PHONE,
        # North American: (555) 123-4567, 555.123.4567, +1 555 123 4567
        re.compile(r"(?<![\d-])(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\d-])"),
    ),
    _Rule(
        EntityType.ADDRESS,
        re.compile(
            # "the 2019 High Court judgment" and "Order 26 County Court Rules"
            # name a court; "3 County Court Road" and "1 Crown Court, London"
            # are addresses.
            r"\b(?:(?P<year>(?:18|19|20)\d\d)\b|(?!(?:18|19|20)\d\d\b)\d{1,4}[A-Za-z]?),?\s+"
            rf"(?!{_COURT_TYPES}\s+(?:Courts?|Tribunals?)\b(?![ \t]+{_STREET_TYPES}\b)"
            r"(?(year)|(?!,[ \t]*[A-Z])))"
            rf"(?:[A-Z][a-z'’\-]+\s+){{1,3}}{_STREET_TYPES}\b"
            # Keep an abbreviation's full stop only mid-sentence ("12 High St. and").
            r"(?:\.(?=\s+[a-z,;]))?"
        ),
    ),
    _Rule(
        EntityType.ADDRESS,
        # UK postcode
        re.compile(r"\b[A-PR-UWYZ][A-HK-Y]?\d[A-HJKPS-UW\d]?\s?\d[ABD-HJLNP-UW-Z]{2}\b"),
    ),
    _Rule(
        EntityType.DATE,
        re.compile(
            rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+\d{{4}}\b"
            rf"|\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
            r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            r"|\b\d{4}-\d{2}-\d{2}\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        EntityType.MONEY,
        re.compile(
            r"(?:[£$€]|\b(?:GBP|USD|EUR)\s?)\d[\d,]*(?:\.\d{1,2})?"
            r"(?:\s?(?:million|billion|thousand|bn|m|k)\b)?"
            r"|\b\d[\d,]*(?:\.\d{1,2})?(?:\s(?:million|billion|thousand))?"
            r"\s?(?:pounds?|dollars?|euros?|GBP|USD|EUR)\b",
            re.IGNORECASE,
        ),
    ),
)

# Capital letters, including accented ones ("É", "Ł").
_CAPITAL = "A-Z" + "".join(c for c in map(chr, range(0xC0, 0x250)) if c.isupper())
# Space within a line, including no-break spaces ("Mr\u00a0Smith") and the
# "\r" of a Windows line break.
_SPACE = r"[^\S\n]"
_ORG_SUFFIX = (
    r"(?:Limited|LIMITED|Ltd|LTD|PLC|plc|Inc|INC|LLC|LLP|L\.L\.P|Corporation|CORPORATION|"
    r"Corp|CORP|Company|COMPANY|Co|CO|GmbH|AG|SA|S\.A|NV|N\.V|BV|B\.V|SE|LP|L\.P)"
)
# A name ends at its first legal form: "Acme Ltd and Beta Ltd" is two companies.
_ORG_SUFFIX_WORD = rf"{_ORG_SUFFIX}(?![\w\-])"  # not "Co-operative"
_ORG_TOKEN = rf"(?!{_ORG_SUFFIX_WORD})(?:[{_CAPITAL}][\w&'’\-]*|&)"
# Only an unambiguous legal form wraps onto the next line ("Acme Trading\nLimited
# and"): a line that starts "Company means" or "Limited Warranty" is not the end
# of a company name.
_WRAPPED_ORG_SUFFIX = (
    r"(?:Limited|LIMITED|Ltd|LTD|PLC|plc|LLP|LLC|Inc|INC|GmbH)(?![\w\-])"
    rf"(?={_SPACE}*(?:\n|\Z|[,.;:)(])|{_SPACE}+[a-z])"
)
_ORG = re.compile(
    rf"\b{_ORG_TOKEN}(?:{_SPACE}+(?:{_ORG_TOKEN}|and|of|the|for|de|du)){{0,6}}"
    rf"(?:{_SPACE}+{_ORG_SUFFIX_WORD}(?:{_SPACE}+{_ORG_SUFFIX_WORD})?"
    rf"|{_SPACE}*\n{_SPACE}*{_WRAPPED_ORG_SUFFIX})"
    # Keep an abbreviation's full stop only mid-sentence ("Acme Ltd. and").
    r"(?:\.(?=\s+[a-z,;]))?"
)

# Words that start a sentence or clause, or describe a role, rather than
# forming part of an organisation's name.
_ORG_LEADING_WORDS = frozenset(
    """
    a after also although an and as at because before between but by dear during
    for from further furthermore however if in it its meanwhile moreover of on our
    re since so that the their then there these this those to under when where
    whereas which while with yesterday today your
    """.split()
)
_JOB_TITLE_WORDS = frozenset(
    """
    board chair chairman chairwoman chief company director directors employee employees
    executive finance general head managing manager member members officer partner
    president secretary shareholder shareholders
    """.split()
)
# Organisation-like names that are public bodies, not private parties.
_PUBLIC_BODY_WORDS = (
    "court",
    "tribunal",
    "parliament",
    "house of lords",
    "privy council",
    "commission",
    "government",
    "ministry",
    "department",
    "crown",
    "secretary of state",
)

# Honorifics that introduce a name, in title case or capitals ("Mr", "MR").
# Abbreviations may take a full stop ("Mr."); words may not, so "Yes, my
# Lord. The claimant..." holds no name.
_ABBREVIATED_TITLES = ("Mr", "Mrs", "Ms", "Mx", "Dr", "Prof", "Rev", "Revd")
_WORD_TITLES = ("Miss", "Professor", "Sir", "Dame", "Lord", "Lady")
_TITLES = _ABBREVIATED_TITLES + _WORD_TITLES
_TITLE_WORDS = frozenset(title.lower() for title in _TITLES)
_ABBREVIATION_WORDS = frozenset(title.lower() for title in _ABBREVIATED_TITLES)


def _title_alternation(titles: Iterable[str]) -> str:
    forms = {form for title in titles for form in (title, title.upper())}
    return "|".join(sorted(forms, key=len, reverse=True))


_TITLE = re.compile(
    rf"(?<![\w'’\-])(?:(?:{_title_alternation(_ABBREVIATED_TITLES)})\.?"
    rf"|{_title_alternation(_WORD_TITLES)})(?![\w'’\-])"
)
# "Dear Sir" and "My Lord" end their line: the next line is not a name.
_SALUTATION = re.compile(rf"(?i:\b(?:dear|my)){_SPACE}+\Z")

# A word of a name: letters with inner apostrophes or hyphens ("O'Brien",
# "Smith-Jones", "José"), without a possessive "'s".
_NAME_WORD = re.compile(r"[^\W\d_]+(?:['’\-](?!s\b)[^\W\d_]+)*")
_NAME_GAP = re.compile(rf"{_SPACE}*\n{_SPACE}*|{_SPACE}+")
# An elision or article before the capital: "d'Souza", "al-Hassan".
_NAME_PREFIX = re.compile(r"^(?:[a-z]{1,2}['’]|(?:al|el)-)")

# Words that end a name: labels ("Witness Statement", "Date:"), job titles
# ("Partner") and legal forms. Legal roles ("Claimant", "Solicitor") are
# added from Anonymiser.LEGAL_PRESERVE. Ordinary words that are also names
# ("Page", "Said", "Lord", "Judge") are deliberately not listed.
_NAME_ENDING_WORDS = frozenset(
    """
    date dated signed signature address tel telephone email fax mobile statement exhibit
    schedule clause section part paragraph partner director associate secretary manager
    chairman chair officer consultant clerk trustee trustees executor executors
    administrator administrators receiver liquidator limited ltd plc llp llc inc company
    co corporation
    """.split()
)
# Post-nominals end a name when not in title case ("Mr John Smith KC", "Lord
# Denning MR"), so the surnames Ma and Sc are still names.
_POST_NOMINALS = frozenset(
    """
    qc kc sc cbe obe mbe kbe dbe gbe mp mep msp jp frcs frcp mrcs mrcp phd dphil llb llm bcl
    mr lj ljj jj cj vc psc dpsc jsc
    """.split()
)
# Function words are never part of a name ("MR SMITH AND MRS JONES", "MR ADAM
# CARTER V DELTA"). Pronouns that are also surnames (He, You, An) are not listed.
_FUNCTION_WORDS = frozenset(
    """
    the this that these those and or nor but if of to in on at as by for from with into
    upon is was are were has had it its we they them their our his my me us v vs re per via
    """.split()
)
# Words that open a line of a letter or list rather than continue a name
# wrapped onto it ("Mr John Smith\nThank you", "Mr John Smith\nHead of Legal").
_LINE_START_WORDS = frozenset(
    """
    dear yours thank thanks please regarding subject further following yes no note
    instructed head
    """.split()
)
# Offices, not names: "Mr Justice Fraser", "Mr Speaker", "Lord Chief Justice",
# "Lord Chancellor", and the quarter day "Lady Day".
_OFFICES = frozenset({"justice", "justices", "speaker", "president"})
_LORD_OFFICES = frozenset(
    """
    chief chancellor advocate president ordinary mayor lieutenant speaker privy high
    provost chamberlain steward commissioner commissioners warden great bishop day
    """.split()
)
_NAME_PARTICLES = frozenset(
    "van von de da di du del della der den la le al el bin ibn ap ter ten dos das".split()
)
# What may follow a name wrapped onto a new line: running text, not a label
# ("Apologies:"), a heading or a company ("Mr John Smith\nAcme Holdings plc").
_WRAPPED_NAME_FOLLOWER = re.compile(
    rf"{_SPACE}*(?:\Z|\()|[,.;!?)\]'’\"”]|{_SPACE}+[—–-]{_SPACE}"
    rf"|{_SPACE}+(?!{_ORG_SUFFIX_WORD})[a-z]"
    rf"|{_SPACE}+(?i:{'|'.join(sorted(_POST_NOMINALS))})\b"
)


@dataclass(frozen=True)
class _NameToken:
    word: str
    start: int
    end: int  # after an initial's full stop
    wrapped: bool  # on a line after the title
    dotted: bool  # followed by a full stop ("J.", "Mr.")


def _name_tokens(text: str, position: int, may_wrap: bool, limit: int = 8) -> list[_NameToken]:
    """The words after a title, up to a gap that cannot fall inside a name."""
    tokens: list[_NameToken] = []
    wrapped = False
    while len(tokens) < limit:
        gap = _NAME_GAP.match(text, position)
        if gap is not None:
            if "\n" in gap.group():
                if wrapped or not may_wrap:
                    break
                wrapped = True
            position = gap.end()
        elif not (tokens and text[tokens[-1].end - 1] == "."):
            break  # words need a space between them, except initials ("J.R. Smith")
        match = _NAME_WORD.match(text, position)
        if match is None:
            break
        end = match.end()
        dotted = text.startswith(".", end)
        if dotted and len(match.group()) == 1:
            end += 1
        tokens.append(_NameToken(match.group(), match.start(), end, wrapped, dotted))
        position = end
    return tokens


def _is_capitalised(word: str) -> bool:
    """True for "Smith", "SMITH", "McDonald", "José", "d'Souza" and "al-Hassan"."""
    return _NAME_PREFIX.sub("", word)[:1].isupper()


def _is_title(word: str) -> bool:
    return word.lower() in _TITLE_WORDS and (word.istitle() or word.isupper())


class Anonymiser:
    """
    Anonymises PII in legal documents while preserving legal structure.

    Features:
    - Consistent replacement (same entity -> same placeholder throughout)
    - Placeholders numbered in reading order: [PERSON_1], [PERSON_2], ...
    - Legal citations are never altered; case names in citations are kept
      when ``preserve_case_names`` is set
    - Structured identifiers: emails, phone numbers, UK postcodes, NI and
      NHS numbers, sort codes, IBANs (checksum validated), payment cards
      (Luhn validated), court claim numbers
    - Reversible with the returned mapping

    Limitations:
        Rule-based detection finds titled names ("Mr Smith", "Dr Jane Doe")
        and companies with a legal-form suffix ("Acme Ltd"). Untitled names
        ("Jane Smith"), and names in scripts without capital letters, need a
        named-entity model: pass ``ner="en_core_web_sm"`` (requires spaCy) or
        your own detector. No automated tool guarantees anonymisation; review
        output before relying on it.

        Case names inside citations are public record, but if a document
        concerns one of the cited cases the preserved names can identify its
        parties. Set ``preserve_case_names=False`` for such documents.

    Example:
        >>> anon = Anonymiser()
        >>> result = anon.anonymise("Mr John Smith of 12 High Street signed the contract")
        >>> print(result.text)
        [PERSON_1] of [ADDRESS_1] signed the contract
        >>> print(result.mapping)
        {'Mr John Smith': '[PERSON_1]', '12 High Street': '[ADDRESS_1]'}
    """

    RULES = RULES
    TITLES = set(_TITLES)

    # Words that mark a titled name as a legal role rather than a person
    # to anonymise ("Mr Justice Smith").
    LEGAL_PRESERVE = {
        "claimant",
        "defendant",
        "appellant",
        "respondent",
        "applicant",
        "plaintiff",
        "petitioner",
        "court",
        "tribunal",
        "judge",
        "justice",
        "barrister",
        "solicitor",
        "counsel",
        "witness",
        "expert",
    }

    def __init__(
        self,
        preserve_case_names: bool = True,
        preserve_dates: bool = False,
        consistent_replacement: bool = True,
        salt: str | None = None,
        entity_types: Iterable[EntityType | str] | None = None,
        ner: str | Detector | None = None,
    ):
        """
        Initialise the anonymiser.

        Args:
            preserve_case_names: Keep party names inside case citations
                ("Smith v Jones [2024] UKSC 1").
            preserve_dates: Don't anonymise dates (useful for legal timelines).
            consistent_replacement: Same entity gets same placeholder.
            salt: Secret for deterministic placeholders such as
                [PERSON_3f9a1c2e]. The same entity then gets the same
                placeholder in every document and every run, without storing
                a mapping. This is pseudonymisation, not anonymisation: anyone
                with the salt can test guesses. Keep it secret.
            entity_types: Entity types to anonymise (EntityType members or
                their values, e.g. "person"). Defaults to all.
            ner: Optional named-entity detector for names the rules miss:
                a spaCy model name such as "en_core_web_sm", or a callable
                returning (start, end, label) spans.
        """
        self.preserve_case_names = preserve_case_names
        self.preserve_dates = preserve_dates
        self.consistent_replacement = consistent_replacement
        self.salt = salt

        if entity_types is None:
            selected = set(EntityType)
        else:
            selected = {_as_entity_type(t) for t in entity_types}
        if preserve_dates:
            selected.discard(EntityType.DATE)
        self.entity_types: frozenset[EntityType] = frozenset(selected)

        self._detector: Detector | None = _spacy_detector(ner) if isinstance(ner, str) else ner
        self._citation_parser = CitationParser()
        # "Judge" and "Justice" are surnames too ("Mrs Sarah Justice"); after a
        # title they are offices, which _titled_name_end handles.
        roles = {word.lower() for word in self.LEGAL_PRESERVE} - {"judge", "justice"}
        self._name_endings = _NAME_ENDING_WORDS | roles | {f"{word}s" for word in roles}
        self._counters: dict[EntityType, int] = {}
        self._lookup: dict[tuple[EntityType, str], str] = {}
        self._mapping: dict[str, str] = {}

    def reset(self):
        """Reset counters and mappings for new document."""
        self._counters = {}
        self._lookup = {}
        self._mapping = {}

    def anonymise(self, text: str, reset: bool = True) -> AnonymisationResult:
        """
        Anonymise PII in text.

        Args:
            text: Text to anonymise
            reset: Reset counters for new document. Pass False to continue
                numbering, and reuse placeholders, from the previous call.

        Returns:
            AnonymisationResult with anonymised text, the entities found
            (offsets into the input text) and the original -> placeholder mapping
        """
        if reset:
            self.reset()

        protected = self._protected_spans(text)
        candidates = self._find_candidates(text)

        # Keep non-overlapping candidates: earliest first, then longest, then
        # by rule priority. Anything touching a citation or preserved case
        # name is dropped.
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[3]))
        accepted: list[tuple[int, int, EntityType, float]] = []
        last_end = -1
        for start, end, entity_type, _, confidence in candidates:
            if start < last_end or any(
                start < p_end and p_start < end for p_start, p_end in protected
            ):
                continue
            accepted.append((start, end, entity_type, confidence))
            last_end = end

        entities = []
        for start, end, entity_type, confidence in accepted:
            original = text[start:end]
            entities.append(
                AnonymisedEntity(
                    original=original,
                    replacement=self._get_replacement(original, entity_type),
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    confidence=confidence,
                )
            )

        parts = []
        position = 0
        for entity in entities:
            parts.append(text[position : entity.start])
            parts.append(entity.replacement)
            position = entity.end
        parts.append(text[position:])

        return AnonymisationResult(
            text="".join(parts), entities=entities, mapping=dict(self._mapping)
        )

    def deanonymise(self, text: str, mapping: dict[str, str]) -> str:
        """
        Reverse anonymisation using mapping.

        Args:
            text: Anonymised text
            mapping: Original -> Replacement mapping

        Returns:
            Original text with PII restored
        """
        reverse: dict[str, str] = {}
        for original, replacement in mapping.items():
            reverse.setdefault(replacement, original)
        if not reverse:
            return text
        pattern = re.compile("|".join(re.escape(r) for r in sorted(reverse, key=len, reverse=True)))
        return pattern.sub(lambda m: reverse[m.group(0)], text)

    def create_training_pair(self, text: str) -> tuple[AnonymisationResult, dict[str, str]]:
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
        reverse_mapping: dict[str, str] = {}
        for original, replacement in result.mapping.items():
            reverse_mapping.setdefault(replacement, original)
        return result, reverse_mapping

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _protected_spans(self, text: str) -> list[tuple[int, int]]:
        """Spans that must not be altered: citations and, optionally, case names."""
        citations = self._citation_parser.parse(text, unique=False)
        spans = [(c.start, c.end) for c in citations if c.start is not None and c.end is not None]
        if self.preserve_case_names:
            names = self._citation_parser.find_case_names(text, citations)
            spans.extend((start, end) for _, start, end in names)
        return spans

    def _find_candidates(self, text: str) -> list[tuple[int, int, EntityType, int, float]]:
        """Return (start, end, type, priority, confidence) for every possible entity."""
        candidates = []
        for priority, rule in enumerate(self.RULES):
            if rule.entity_type not in self.entity_types:
                continue
            for match in rule.pattern.finditer(text):
                group = "value" if "value" in rule.pattern.groupindex else 0
                start, end = match.span(group)
                if rule.validator is not None and not rule.validator(match.group(group)):
                    continue
                candidates.append((start, end, rule.entity_type, priority, 1.0))

        low_priority = len(self.RULES)
        if EntityType.PERSON in self.entity_types:
            for start, end in self._find_person_names(text):
                candidates.append((start, end, EntityType.PERSON, low_priority, 0.9))
        if EntityType.ORGANISATION in self.entity_types:
            for start, end in self._find_organisations(text):
                candidates.append((start, end, EntityType.ORGANISATION, low_priority + 1, 0.9))
        if self._detector is not None:
            for start, end, label in self._detector(text):
                entity_type = _entity_type_for_label(label)
                if entity_type is None or entity_type not in self.entity_types:
                    continue
                if entity_type is EntityType.ORGANISATION and _is_public_body(text[start:end]):
                    continue
                candidates.append((start, end, entity_type, low_priority + 2, 0.8))
        return candidates

    def _find_person_names(self, text: str) -> list[tuple[int, int]]:
        """
        Find titled person names ("Mr Smith", "Dr Jane Doe", "MR ADAM CARTER").

        A name ends at a role or label ("Mr Adam Carter\nClaimant"), a
        function word, or the title of the next name. A title followed by
        an office ("Mr Justice Fraser", "Lord Chancellor") is not a name.
        """
        spans: list[tuple[int, int]] = []
        for title in _TITLE.finditer(text):
            if spans and title.start() < spans[-1][1]:
                continue  # inside the previous name ("Professor Sir John Smith")
            end = self._titled_name_end(text, title)
            if end is not None:
                spans.append((title.start(), end))
        return spans

    def _titled_name_end(self, text: str, title: re.Match) -> int | None:
        """Where the name after a title ends, or None if no name follows it."""
        caps = title.group().rstrip(".").isupper()
        # A name in capitals, or after "Dear Sir" or "My Lord", stays on its line.
        salutation = _SALUTATION.search(text, max(0, title.start() - 12), title.start())
        tokens = _name_tokens(text, title.end(), may_wrap=not caps and salutation is None)

        index = 0
        while index < len(tokens) and self._ends_name(tokens, index, caps):
            index += 1  # further titles: "Professor Sir John Smith"
        last_title = tokens[index - 1].word if index else title.group()
        name: list[_NameToken] = []
        words = 0
        for position in range(index, len(tokens)):
            kind = self._name_part(tokens, position, caps, bool(name))
            if kind is None or (kind == "word" and words == 4):
                break
            words += kind == "word"
            name.append(tokens[position])

        name = _trim_name(name)
        if any(token.wrapped for token in name) and not _WRAPPED_NAME_FOLLOWER.match(
            text, name[-1].end
        ):
            # Not running text after the next line's words: keep the first line.
            name = _trim_name([token for token in name if not token.wrapped])
        if not name:
            return None
        first = name[0].word.lower()
        if first in _OFFICES or (last_title.lower() in ("lord", "lady") and first in _LORD_OFFICES):
            return None
        return name[-1].end

    def _ends_name(self, tokens: list[_NameToken], index: int, caps: bool) -> bool:
        """True if tokens[index] is a title that starts another name."""
        token = tokens[index]
        if not _is_title(token.word):
            return False
        if token.dotted and token.word.lower() in _ABBREVIATION_WORDS:
            return True  # "Mrs Jones Mr. Smith", but not "Mr Peter Lord."
        return index + 1 < len(tokens) and (
            self._name_part(tokens, index + 1, caps, False) is not None
        )

    def _name_part(
        self, tokens: list[_NameToken], index: int, caps: bool, continues: bool
    ) -> str | None:
        """
        Classify tokens[index] as part of a name: "initial", "particle" or
        "word", or None if the name has ended.

        Args:
            continues: The token follows another part of the same name.
        """
        token = tokens[index]
        word = token.word
        lower = word.lower()
        if token.wrapped and lower in _LINE_START_WORDS:
            return None
        if len(word) == 1:
            if not word.isupper() or (continues and word == "V"):
                return None  # "MR ADAM CARTER V DELTA"
            return "initial"
        if lower in _FUNCTION_WORDS or lower in self._name_endings:
            return None
        if continues and lower in _POST_NOMINALS and not word.istitle():
            return None
        if continues and self._ends_name(tokens, index, caps):
            return None  # "MR JOHN SMITH MRS JANE DOE"
        if lower in _NAME_PARTICLES and word.islower():
            return "particle"  # "Mr van der Berg"; must precede a word
        if caps:
            return "word" if word.isupper() else None
        return "word" if _is_capitalised(word) else None

    def _find_organisations(self, text: str) -> list[tuple[int, int]]:
        """Find company names ending in a legal form ("Acme Trading Ltd")."""
        spans = []
        for match in _ORG.finditer(text):
            tokens = list(re.finditer(r"\S+", match.group(0)))
            words = [t.group(0) for t in tokens]
            first = 0
            # "Mr Brown and Acme Ltd", "Mr Smith of Acme Ltd": a person comes
            # first, and the organisation starts after the linking word.
            titles = [i for i, w in enumerate(words) if _is_title(w.rstrip("."))]
            if titles:
                links = [
                    i
                    for i, w in enumerate(words)
                    if i > titles[-1] and w.lower() in ("and", "of", "&")
                ]
                if links:
                    first = links[0] + 1
            # "MR ADAM CARTER V DELTA FREIGHT LIMITED": the company follows "v".
            versus = [i for i, w in enumerate(words) if w.lower().rstrip(".") in ("v", "vs")]
            if versus:
                first = max(first, versus[-1] + 1)
            # Drop sentence starters ("Yesterday Acme Ltd").
            while first < len(words) - 1 and words[first].lower() in _ORG_LEADING_WORDS:
                first += 1
            # Drop a role before "of" ("Managing Director of Acme Ltd").
            if "of" in words[first:]:
                of_index = words.index("of", first)
                if all(w.lower() in _JOB_TITLE_WORDS for w in words[first:of_index]):
                    first = of_index + 1
            if first >= len(words) - 1:
                continue
            start = match.start() + tokens[first].start()
            if _is_public_body(text[start : match.end()]):
                continue
            spans.append((start, match.end()))
        return spans

    def _get_replacement(self, original: str, entity_type: EntityType) -> str:
        """Get or create replacement for entity."""
        key = (entity_type, re.sub(r"\s+", " ", original).strip().lower())

        if self.consistent_replacement and key in self._lookup:
            replacement = self._lookup[key]
        elif self.salt is not None:
            digest = hashlib.sha256(f"{self.salt}|{key[0].value}|{key[1]}".encode()).hexdigest()
            replacement = f"[{entity_type.name}_{digest[:8]}]"
        else:
            self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
            replacement = f"[{entity_type.name}_{self._counters[entity_type]}]"

        if self.consistent_replacement or self.salt is not None:
            self._lookup[key] = replacement
            self._mapping.setdefault(original, replacement)
        return replacement


def _trim_name(name: list[_NameToken]) -> list[_NameToken]:
    """Drop trailing initials and particles: "Mr A", "My Lady I am", "Mr Smith de facto"."""
    name = list(name)
    while name and (len(name[-1].word) == 1 or name[-1].word.islower()):
        name.pop()
    return name


def _as_entity_type(value: EntityType | str) -> EntityType:
    """Accept an EntityType, its value ("person") or its name ("PERSON")."""
    if isinstance(value, EntityType):
        return value
    try:
        return EntityType(value.lower())
    except ValueError:
        valid = ", ".join(t.value for t in EntityType)
        raise ValueError(f"Unknown entity type: {value!r}. Valid types: {valid}") from None


def _entity_type_for_label(label: str) -> EntityType | None:
    if label.upper() in NER_LABELS:
        return NER_LABELS[label.upper()]
    try:
        return EntityType(label.lower())
    except ValueError:
        return None


def _is_public_body(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _PUBLIC_BODY_WORDS)


def _spacy_detector(model_name: str) -> Detector:
    """Build a detector from a spaCy pipeline."""
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "NER-based anonymisation needs spaCy: pip install 'legal-llm-toolkit[ner]' "
            f"and python -m spacy download {model_name}"
        ) from e
    nlp = spacy.load(model_name)

    def detect(text: str) -> list[tuple[int, int, str]]:
        return [(ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents]

    return detect
