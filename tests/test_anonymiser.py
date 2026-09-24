"""
Tests for PII anonymisation.
"""

import pytest

from legalkit.preprocess import Anonymiser
from legalkit.preprocess.anonymiser import EntityType


@pytest.fixture
def anon():
    return Anonymiser()


def anonymised(text, **kwargs):
    return Anonymiser(**kwargs).anonymise(text).text


class TestEntities:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Contact john.smith@example.com for details", "Contact [EMAIL_1] for details"),
            ("Call us on 07123 456789", "Call us on [PHONE_1]"),
            ("Call +44 (0)20 7123 4567 today", "Call [PHONE_1] today"),
            ("Call (555) 123-4567", "Call [PHONE_1]"),
            (
                "Write to 12 High Street, London SW1A 1AA",
                "Write to [ADDRESS_1], London [ADDRESS_2]",
            ),
            ("Sort code 12-34-56", "Sort code [ACCOUNT_NUMBER_1]"),
            ("account number 12345678", "account number [ACCOUNT_NUMBER_1]"),
            ("IBAN GB82 WEST 1234 5698 7654 32", "IBAN [ACCOUNT_NUMBER_1]"),
            ("card 4111 1111 1111 1111", "card [ACCOUNT_NUMBER_1]"),
            ("NI number AB 12 34 56 C", "NI number [NATIONAL_ID_1]"),
            ("NHS number 943 476 5919", "NHS number [NATIONAL_ID_1]"),
            ("SSN 123-45-6789", "SSN [NATIONAL_ID_1]"),
            ("Claim No. HC-2014-000123", "Claim No. [CASE_NUMBER_1]"),
            ("on 15 January 2024", "on [DATE_1]"),
            ("on March 3, 2024", "on [DATE_1]"),
            ("on 2024-01-15", "on [DATE_1]"),
            ("for £150,000", "for [MONEY_1]"),
            ("for USD 2,000.50", "for [MONEY_1]"),
            ("for 3 million dollars", "for [MONEY_1]"),
        ],
    )
    def test_structured_identifiers(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "IBAN GB82 WEST 1234 5698 7654 33",  # bad check digits
            "card 4111 1111 1111 1112",  # fails Luhn
        ],
    )
    def test_invalid_identifiers_are_left_alone(self, text):
        assert anonymised(text) == text

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Mr Smith said so.", "[PERSON_1] said so."),
            ("Dr Jane Doe attended.", "[PERSON_1] attended."),
            ("Mr J. Smith attended.", "[PERSON_1] attended."),
            ("Mrs O'Brien's evidence", "[PERSON_1]'s evidence"),
            ("Ms Smith-Jones replied.", "[PERSON_1] replied."),
            ("Mr Smith QC appeared.", "[PERSON_1] QC appeared."),
            ("Mr Courtney Smith attended.", "[PERSON_1] attended."),
        ],
    )
    def test_titled_names(self, text, expected):
        assert anonymised(text) == expected

    def test_judicial_titles_are_kept(self):
        text = "Mr Justice Fraser and Lord Justice Leggatt presided."
        assert anonymised(text) == text

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "I write to confirm that ABC Limited has instructed us.",
                "I write to confirm that [ORGANISATION_1] has instructed us.",
            ),
            ("Yesterday Acme Trading Ltd paid.", "Yesterday [ORGANISATION_1] paid."),
            (
                "the Managing Director of Beta Holdings plc",
                "the Managing Director of [ORGANISATION_1]",
            ),
            ("Mr Brown and Gamma LLP.", "[PERSON_1] and [ORGANISATION_1]."),
            ("Acme Ltd. and others", "[ORGANISATION_1] and others"),
            ("The Royal Bank of Scotland plc lent", "The [ORGANISATION_1] lent"),
        ],
    )
    def test_organisations(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "As stated at paragraph 2 of the judgment of the Court, costs follow.",
            "The claims 3 of the way forward were close.",
            "Clause 1.2.10 applies.",
        ],
    )
    def test_prose_is_not_anonymised(self, text):
        assert anonymised(text) == text


class TestLegalStructure:
    def test_citations_are_never_altered(self):
        text = "See [2024] UKSC 15 and 347 U.S. 483 and Case C-6/90."
        assert anonymised(text, preserve_case_names=False) == text

    def test_case_names_preserved(self):
        text = "Caparo Industries plc v Dickman [1990] 2 AC 605 applies to Mr Smith."
        assert anonymised(text) == (
            "Caparo Industries plc v Dickman [1990] 2 AC 605 applies to [PERSON_1]."
        )

    def test_case_names_not_preserved(self):
        text = "Caparo Industries plc v Dickman [1990] 2 AC 605"
        assert anonymised(text, preserve_case_names=False) == (
            "[ORGANISATION_1] v Dickman [1990] 2 AC 605"
        )

    def test_preserve_dates(self):
        assert anonymised("on 15 January 2024", preserve_dates=True) == "on 15 January 2024"


class TestReplacement:
    def test_consistent_replacement(self, anon):
        result = anon.anonymise("Mr Smith said. Later, Mr Smith confirmed.")
        assert result.text == "[PERSON_1] said. Later, [PERSON_1] confirmed."
        assert result.mapping == {"Mr Smith": "[PERSON_1]"}

    def test_numbering_follows_reading_order(self, anon):
        result = anon.anonymise("Dr Jones emailed a@b.com and then Mr Smith emailed c@d.com.")
        assert result.text == (
            "[PERSON_1] emailed [EMAIL_1] and then [PERSON_2] emailed [EMAIL_2]."
        )

    def test_inconsistent_replacement(self):
        result = Anonymiser(consistent_replacement=False).anonymise("Mr Smith and Mr Smith")
        assert result.text == "[PERSON_1] and [PERSON_2]"
        assert result.mapping == {}

    def test_case_insensitive_matching_keeps_first_surface_form(self, anon):
        result = anon.anonymise("John@Example.com and john@example.com")
        assert result.text == "[EMAIL_1] and [EMAIL_1]"
        assert result.mapping == {"John@Example.com": "[EMAIL_1]", "john@example.com": "[EMAIL_1]"}

    def test_overlapping_entities_do_not_corrupt_text(self, anon):
        # A sort code also looks like a date; only one replacement may apply.
        assert anon.anonymise("code 12-34-56 end").text == "code [ACCOUNT_NUMBER_1] end"

    def test_entities_report_offsets_into_input(self, anon):
        text = "Mr Smith paid £5 on 1 May 2024."
        result = anon.anonymise(text)
        assert [e.entity_type for e in result.entities] == [
            EntityType.PERSON,
            EntityType.MONEY,
            EntityType.DATE,
        ]
        for entity in result.entities:
            assert text[entity.start : entity.end] == entity.original

    def test_reset_false_continues_numbering(self, anon):
        anon.anonymise("Mr Smith")
        assert anon.anonymise("Mr Jones and Mr Smith", reset=False).text == (
            "[PERSON_2] and [PERSON_1]"
        )

    def test_salted_placeholders_are_stable_across_documents(self):
        first = Anonymiser(salt="secret").anonymise("Mr Smith").text
        second = Anonymiser(salt="secret").anonymise("Mr Jones met Mr Smith").text
        assert first.startswith("[PERSON_") and first != "[PERSON_1]"
        assert second.endswith(first)
        assert Anonymiser(salt="other").anonymise("Mr Smith").text != first

    def test_entity_type_selection(self):
        text = "Mr Smith emailed a@b.com"
        assert anonymised(text, entity_types=["email"]) == "Mr Smith emailed [EMAIL_1]"
        assert anonymised(text, entity_types=[EntityType.PERSON]) == "[PERSON_1] emailed a@b.com"
        assert anonymised(text, entity_types=["PERSON"]) == "[PERSON_1] emailed a@b.com"
        with pytest.raises(ValueError, match="Unknown entity type"):
            Anonymiser(entity_types=["pets"])


class TestReversal:
    def test_deanonymise_round_trip(self, anon):
        original = "Mr Smith (john@test.com) paid £5 to Mr Jones on 1 May 2024."
        result = anon.anonymise(original)
        assert anon.deanonymise(result.text, result.mapping) == original

    def test_deanonymise_does_not_confuse_similar_placeholders(self, anon):
        mapping = {"a": "[PERSON_1]", "b": "[PERSON_10]"}
        assert anon.deanonymise("[PERSON_10] [PERSON_1]", mapping) == "b a"

    def test_create_training_pair(self, anon):
        result, reverse = anon.create_training_pair("Email x@y.com")
        assert result.text == "Email [EMAIL_1]"
        assert reverse == {"[EMAIL_1]": "x@y.com"}
        assert result.get_original("[EMAIL_1]") == "x@y.com"


class TestNamedEntityDetector:
    def test_custom_detector_finds_untitled_names(self):
        def detector(text):
            start = text.find("Jane Smith")
            return [(start, start + len("Jane Smith"), "PERSON")] if start >= 0 else []

        result = Anonymiser(ner=detector).anonymise("Signed, Jane Smith, Partner")
        assert result.text == "Signed, [PERSON_1], Partner"

    def test_detector_labels_and_public_bodies(self):
        def detector(text):
            return [
                (0, 11, "ORG"),  # "Acme Widget"
                (16, 29, "ORG"),  # "Supreme Court"
                (34, 38, "GPE"),  # unsupported label
            ]

        text = "Acme Widget and Supreme Court and Kent"
        assert Anonymiser(ner=detector).anonymise(text).text == (
            "[ORGANISATION_1] and Supreme Court and Kent"
        )

    def test_spacy_model(self, tmp_path):
        spacy = pytest.importorskip("spacy")
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns([{"label": "PERSON", "pattern": "Jane Smith"}])
        nlp.to_disk(tmp_path / "model")

        result = Anonymiser(ner=str(tmp_path / "model")).anonymise("Signed, Jane Smith")
        assert result.text == "Signed, [PERSON_1]"
