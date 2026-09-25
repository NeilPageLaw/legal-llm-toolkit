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
            rf"\b\d{{1,4}}[A-Za-z]?,?\s+(?!{_COURT_TYPES}\s+(?:Courts?|Tribunal)\b)"
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

_TITLES = (
    "Mr",
    "Mrs",
    "Ms",
    "Miss",
    "Mx",
    "Dr",
    "Prof",
    "Professor",
    "Sir",
    "Dame",
    "Lord",
    "Lady",
    "Rev",
    "Revd",
)
_TITLE_ALTERNATION = "|".join(sorted(_TITLES, key=len, reverse=True))

# Words that follow a name without being part of it: headings and labels
# ("Mr John Smith\nDate: ..."), roles ("Mr Adam Carter\nClaimant"), job
# titles and legal forms. Legal roles are added from Anonymiser.LEGAL_PRESERVE.
_NAME_STOPWORDS = frozenset(
    """
    the this that these those a an and or but if in on at to for of by with from as is was
    has had said who vs date dated signed signature address tel telephone email fax mobile
    re dear yours thank thanks page statement exhibit schedule clause section part paragraph
    partner director associate secretary manager chairman chair officer consultant clerk
    trustee trustees executor executors administrator administrators receiver liquidator
    limited ltd plc llp llc inc company co corporation
    """.split()
) | {title.lower() for title in _TITLES}


def _titled_name_patterns(role_words: Iterable[str]) -> tuple[re.Pattern, re.Pattern]:
    """Patterns for titled names in normal case and in capitals."""
    stopwords = (
        _NAME_STOPWORDS | {w.lower() for w in role_words} | {f"{w.lower()}s" for w in role_words}
    )
    stop = rf"(?!(?i:{'|'.join(sorted(stopwords, key=len, reverse=True))})\b)"

    # A capitalised name ("Smith", "O'Brien", "McDonald", "Smith-Jones") or an initial.
    token = rf"{stop}(?:(?:[A-Z]['’])?[A-Z][A-Za-z\-]*[a-z]|[A-Z](?:\.|\b(?!['’])))"
    # A name stays on its line, except that one word may wrap onto the next
    # line when running text continues after it ("Mr John\nSmith said").
    wrapped = rf"(?:[ \t]*\n[ \t]*{token}(?=[ \t]+[a-z]|[,.;:)\]'’]))?"
    titled = re.compile(
        rf"\b(?:{_TITLE_ALTERNATION})\.?[ \t]+(?P<name>{token}(?:[ \t]+{token}){{0,3}}{wrapped})"
    )

    # Names in capitals, as in the heading of a judgment ("MR ADAM CARTER").
    upper_token = rf"{stop}[A-Z][A-Z'’\-]*[A-Z]"
    upper = re.compile(
        rf"\b(?i:{_TITLE_ALTERNATION})\.?[ \t]+"
        rf"(?P<name>(?:[A-Z]\.?[ \t]+){{0,2}}{upper_token}(?:[ \t]+{upper_token}){{0,2}})\b"
    )
    return titled, upper


_ORG_SUFFIX = (
    r"(?:Limited|LIMITED|Ltd|LTD|PLC|plc|Inc|INC|LLC|LLP|L\.L\.P|Corporation|CORPORATION|"
    r"Corp|CORP|Company|COMPANY|Co|CO|GmbH|AG|SA|S\.A|NV|N\.V|BV|B\.V|SE|LP|L\.P)"
)
# A name ends at its first legal form ("Acme Ltd and Beta Ltd" is two
# companies) and stays on one line.
_ORG_TOKEN = rf"(?!{_ORG_SUFFIX}\b)(?:[A-Z][\w&'’\-]*|&)"
_ORG = re.compile(
    rf"\b{_ORG_TOKEN}(?:[ \t]+(?:{_ORG_TOKEN}|and|of|the|for|de|du)){{0,6}}"
    rf"[ \t]+{_ORG_SUFFIX}\b(?:[ \t]+{_ORG_SUFFIX}\b)?"
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
        ("Jane Smith") need a named-entity model: pass ``ner="en_core_web_sm"``
        (requires spaCy) or your own detector. No automated tool guarantees
        anonymisation; review output before relying on it.

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
        self._name_patterns = _titled_name_patterns(self.LEGAL_PRESERVE)
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

        Role words end a name ("Mr Adam Carter\nClaimant"), and a title
        followed by one ("Mr Justice Fraser") is not a name at all.
        """
        return [match.span() for pattern in self._name_patterns for match in pattern.finditer(text)]

    def _find_organisations(self, text: str) -> list[tuple[int, int]]:
        """Find company names ending in a legal form ("Acme Trading Ltd")."""
        spans = []
        for match in _ORG.finditer(text):
            tokens = list(re.finditer(r"\S+", match.group(0)))
            words = [t.group(0) for t in tokens]
            first = 0
            # "Mr Brown and Acme Ltd", "Mr Smith of Acme Ltd": a person comes
            # first, and the organisation starts after the linking word.
            titles = [i for i, w in enumerate(words) if w.rstrip(".").title() in self.TITLES]
            if titles:
                links = [
                    i for i, w in enumerate(words) if i > titles[-1] and w in ("and", "of", "&")
                ]
                if not links:
                    continue
                first = links[-1] + 1
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
