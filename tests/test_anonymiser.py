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

    def test_names_in_capitals_as_in_judgment_headings(self):
        text = "BETWEEN:\nMR ADAM CARTER (Claimant)\nand\nDELTA LOGISTICS LTD (Defendant)\n\nMr Adam Carter says"
        assert anonymised(text) == (
            "BETWEEN:\n[PERSON_1] (Claimant)\nand\n[ORGANISATION_1] (Defendant)\n\n[PERSON_1] says"
        )

    def test_names_in_capitals_stop_at_conjunctions(self):
        assert anonymised("MR SMITH AND MRS SMITH v ACME LIMITED") == (
            "[PERSON_1] AND [PERSON_2] v [ORGANISATION_1]"
        )
        assert anonymised("MR JUSTICE FRASER:") == "MR JUSTICE FRASER:"

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


class TestLayouts:
    """Regressions from review: layouts common in legal documents."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "BETWEEN:\nMr Adam Carter\nClaimant\nand\nACME TRADING LIMITED\nDefendant",
                "BETWEEN:\n[PERSON_1]\nClaimant\nand\n[ORGANISATION_1]\nDefendant",
            ),
            (
                "Signed: Mr John Smith\nSolicitor for the Claimant",
                "Signed: [PERSON_1]\nSolicitor for the Claimant",
            ),
            (
                "Exhibit 3: Mrs Jane Jones Witness Statement",
                "Exhibit 3: [PERSON_1] Witness Statement",
            ),
            ("Witness: Mr John Smith\nDate: 1 May 2020", "Witness: [PERSON_1]\nDate: [DATE_1]"),
            ("Mr John Smith\nThe court held that", "[PERSON_1]\nThe court held that"),
            ("the evidence of Mr John\nSmith, who said", "the evidence of [PERSON_1], who said"),
            ("MR JOHN SMITH SOLICITOR", "[PERSON_1] SOLICITOR"),
        ],
    )
    def test_names_stop_at_roles_labels_and_line_ends(self, text, expected):
        assert anonymised(text) == expected

    def test_separate_companies_stay_separate(self):
        text = "Acme Trading Ltd and Beta Services Ltd entered into the Agreement."
        assert (
            anonymised(text) == "[ORGANISATION_1] and [ORGANISATION_2] entered into the Agreement."
        )
        block = "ACME TRADING LIMITED\nClaimant\nand\nBETA SERVICES LIMITED\nDefendant"
        assert anonymised(block) == "[ORGANISATION_1]\nClaimant\nand\n[ORGANISATION_2]\nDefendant"

    def test_person_of_company(self):
        assert anonymised("Mr Smith of Acme Holdings Ltd wrote.") == (
            "[PERSON_1] of [ORGANISATION_1] wrote."
        )

    def test_court_names_are_not_addresses(self):
        text = "In 2019 High Court proceedings were issued before 5 Crown Court judges."
        assert anonymised(text) == text
        assert anonymised("She lives at 12 Maple Court.") == "She lives at [ADDRESS_1]."

    def test_wrapped_iban_does_not_crash(self):
        text = "IBAN GB82 WEST 1234\n5698 7654 32 or GB82\tWEST 1234 5698 7654 32"
        assert anonymised(text) == "IBAN [ACCOUNT_NUMBER_1] or [ACCOUNT_NUMBER_1]"

    def test_sentence_full_stops_are_kept(self):
        assert anonymised("Send it to 12 High St. and wait.") == "Send it to [ADDRESS_1] and wait."
        assert anonymised("It went to 12 High St.") == "It went to [ADDRESS_1]."


class TestSecondReviewRegressions:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Mr A Smith gave evidence.", "[PERSON_1] gave evidence."),
            ("Mrs A. Jones signed the lease.", "[PERSON_1] signed the lease."),
            ("Mr John A Smith", "[PERSON_1]"),
            ("Mr Neil Page", "[PERSON_1]"),
            ("Mr Page said so.", "[PERSON_1] said so."),
            ("Mr Said Ahmed", "[PERSON_1]"),
            ("Ms An Nguyen", "[PERSON_1]"),
            ("Mr Peter Lord", "[PERSON_1]"),
        ],
    )
    def test_names_that_are_also_common_words(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "the evidence given by Mr\nJohn Smith was accepted.",
                "the evidence given by [PERSON_1] was accepted.",
            ),
            ("Mr John\nSmith QC", "[PERSON_1] QC"),
            ("Mr John\nSmith (the Claimant)", "[PERSON_1] (the Claimant)"),
            ("Mr John\nPaul Smith said so.", "[PERSON_1] said so."),
            (
                "the agreement between Acme Trading\nLimited and the tenant",
                "the agreement between [ORGANISATION_1] and the tenant",
            ),
            ("Mr John Smith\nAcme LLP", "[PERSON_1]\n[ORGANISATION_1]"),
        ],
    )
    def test_wrapped_names(self, text, expected):
        assert anonymised(text) == expected

    def test_wrapped_names_through_the_preprocessor(self):
        from legalkit.preprocess import LegalPreprocessor

        result = LegalPreprocessor(anonymise=True).process("given by Mr\nJohn Smith was accepted.")
        assert result.processed == "given by [PERSON_1] was accepted."

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "Mr Smith of Royal Bank of Scotland plc wrote.",
                "[PERSON_1] of [ORGANISATION_1] wrote.",
            ),
            ("Mr Brown of Marks & Spencer plc wrote.", "[PERSON_1] of [ORGANISATION_1] wrote."),
            ("MS Amlin plc insured the risk.", "[ORGANISATION_1] insured the risk."),
            ("DR Horton Inc built homes.", "[ORGANISATION_1] built homes."),
            ("Little Miss Sunshine Ltd", "[ORGANISATION_1]"),
            ("The Co-operative Bank plc lent the money.", "The [ORGANISATION_1] lent the money."),
        ],
    )
    def test_organisation_boundaries(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "The Lord Chief Justice gave judgment.",
            "LORD CHIEF JUSTICE",
            "Lady Chief Justice Carr presiding.",
        ],
    )
    def test_judicial_offices_are_kept(self, text):
        assert anonymised(text) == text

    def test_streets_named_after_courts_are_addresses(self):
        assert (
            anonymised("She lives at 1 Crown Court, London.") == "She lives at [ADDRESS_1], London."
        )
        assert anonymised("3 County Court Road") == "[ADDRESS_1]"


class TestThirdReviewRegressions:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("MR JOHN SMITH MRS JANE DOE", "[PERSON_1] [PERSON_2]"),
            ("Mrs Ann Lee\nMr Bob Cole signed it.", "[PERSON_1]\n[PERSON_2] signed it."),
            ("Mrs Jones Mr. Smith", "[PERSON_1] [PERSON_2]"),
            ("Professor Sir John Smith said", "[PERSON_1] said"),
            ("I spoke to Mr Peter Lord.", "I spoke to [PERSON_1]."),
            ("Lord Denning MR said", "[PERSON_1] MR said"),
            ("MR ADAM CARTER KC", "[PERSON_1] KC"),
        ],
    )
    def test_where_names_end(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "MR WEI HE",
            "MRS LI YOU",
            "MR NEIL PAGE",
            "MS AN NGUYEN",
            "Mr José Álvarez",
            "Ms Zoë Müller",
            "Mr Łukasz Nowak",
            "Mr John SMITH",
            "Mr de Souza",
            "Mr van der Berg",
            "Mr d’Souza",
            "Mr al-Hassan",
            "Mr Justin Judge",
            "Mrs Sarah Justice",
            "Mr Ma",
            "Mr John Smith",
            "Mr John\r\nSmith",
        ],
    )
    def test_whole_name_is_anonymised(self, text):
        assert anonymised(text) == "[PERSON_1]"

    @pytest.mark.parametrize(
        "text",
        [
            "Dear Sir\nThank you for your letter.",
            "Dear Sir\nWe act for the tenant.",
            "Dear Sir\nRe: Lease of 12 Park Road",
            "Dear Sir\nPlease find enclosed the lease.",
            "It was noted by Mr\nThe court then rose.",
            "Yes, my Lord. The claimant accepts that.",
            "My Lady I am grateful.",
            "The Lord Chancellor may make rules.",
            "Rent is due on Lady Day.",
            "Mr Speaker, I beg to move.",
            "The Lord Mayor of London attended.",
            "The hearing was held by MS Teams.",
        ],
    )
    def test_salutations_and_offices_are_not_names(self, text):
        # "Re: Lease of 12 Park Road" holds an address; nothing else changes.
        assert anonymised(text, entity_types=["person"]) == text

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Mr John Smith\nAcme Holdings plc", "[PERSON_1]\n[ORGANISATION_1]"),
            (
                "Present: Mr John Smith\nApologies: Mrs Jane Doe",
                "Present: [PERSON_1]\nApologies: [PERSON_2]",
            ),
            ("Mr John Smith\nHead of Legal", "[PERSON_1]\nHead of Legal"),
            ("MR SMITH OF DELTA LOGISTICS LTD", "[PERSON_1] OF [ORGANISATION_1]"),
            ("MR ADAM CARTER V DELTA FREIGHT LIMITED", "[PERSON_1] V [ORGANISATION_1]"),
            ("Électricité de France SA agreed.", "[ORGANISATION_1] agreed."),
            ("Acme Trading Ltd agreed.", "[ORGANISATION_1] agreed."),
            ("Acme Trading\r\nLimited and the tenant", "[ORGANISATION_1] and the tenant"),
        ],
    )
    def test_people_and_companies_on_neighbouring_lines(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Director\nCompany Secretary",
            "Definitions\nCompany means Acme.",
            "12. WARRANTY\nLimited Warranty. The Supplier warrants",
        ],
    )
    def test_labels_are_not_companies(self, text):
        assert anonymised(text) == text

    @pytest.mark.parametrize(
        "text",
        [
            "the 2019 High Court\nproceedings",
            "a 2019 County Court Judgment",
            "the 2019 High Court (Chancery Division)",
            "the 2019 High Court, Court of Appeal and Supreme Court decisions",
            "Order 26 County Court Rules",
        ],
    )
    def test_courts_after_numbers_are_not_addresses(self, text):
        assert anonymised(text) == text


class TestFourthReviewRegressions:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("He wrote: 'Mr Smith has not paid.'", "He wrote: '[PERSON_1] has not paid.'"),
            ("the defendant ('Mr Smith')", "the defendant ('[PERSON_1]')"),
            ("Claimant-Mr Smith", "Claimant-[PERSON_1]"),
            ("Mr John V. Smith gave evidence.", "[PERSON_1] gave evidence."),
            ("Mr Per Svensson gave evidence.", "[PERSON_1] gave evidence."),
            ("Mr To Kwan-hang attended.", "[PERSON_1] attended."),
            ("Ms Or Cohen signed.", "[PERSON_1] signed."),
            ("Mr Minh To said so.", "[PERSON_1] said so."),
            ("Mr Smith In Person", "[PERSON_1] In Person"),
            ("mr SMITH attended", "[PERSON_1] attended"),
            ("Mr Peter Lord QC appeared.", "[PERSON_1] QC appeared."),
            ("Mr David Lord MP spoke.", "[PERSON_1] MP spoke."),
        ],
    )
    def test_names(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "Cross-examined by Mr\nSmith: Did you sign it?",
                "Cross-examined by [PERSON_1]: Did you sign it?",
            ),
            ("by Dr\nPaul J. Brown\nClaimant", "by [PERSON_1]\nClaimant"),
            ("Evidence of Mr John\nSmith: he denied it.", "Evidence of [PERSON_1]: he denied it."),
            (
                "I spoke to Dr Sarah\nPage who examined me.",
                "I spoke to [PERSON_1] who examined me.",
            ),
            (
                "A letter from Mr Smith.\nMr Smith\nHe replied the next day.",
                "A letter from [PERSON_1].\n[PERSON_1]\nHe replied the next day.",
            ),
        ],
    )
    def test_names_across_lines(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "The Hongkong and Shanghai Banking\nCorporation Limited lent the money.",
                "The [ORGANISATION_1] lent the money.",
            ),
            (
                "The contract with Deutsche Handels\nAG was terminated.",
                "The contract with [ORGANISATION_1] was terminated.",
            ),
            ("Philips Lighting\nN.V. and Acme", "[ORGANISATION_1] and Acme"),
            ("Acme Trading\nLimited Company number 12345", "[ORGANISATION_1] number 12345"),
            ('Acme Trading\nLimited "the Seller"', '[ORGANISATION_1] "the Seller"'),
            ("Henry V Ltd supplied the goods.", "[ORGANISATION_1] supplied the goods."),
            ("Class V Holdings Ltd agreed.", "[ORGANISATION_1] agreed."),
            ("LORD AND TAYLOR LLC", "[ORGANISATION_1]"),
            ("LORD & TAYLOR LLC", "[ORGANISATION_1]"),
        ],
    )
    def test_organisations(self, text, expected):
        assert anonymised(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Mr John Smith\n1 Crown Court\nLondon", "[PERSON_1]\n[ADDRESS_1]\nLondon"),
            ("She lives at 1 Crown Court.", "She lives at [ADDRESS_1]."),
            ("22 Admiralty Court (rear entrance)", "[ADDRESS_1] (rear entrance)"),
            ("before 5 Crown Court judges", "before 5 Crown Court judges"),
        ],
    )
    def test_court_named_buildings(self, text, expected):
        assert anonymised(text) == expected

    def test_titles_can_be_customised(self):
        class WithRabbi(Anonymiser):
            TITLES = Anonymiser.TITLES | {"Rabbi"}

        class WithoutLord(Anonymiser):
            TITLES = Anonymiser.TITLES - {"Lord"}

        assert WithRabbi().anonymise("Rabbi Jonathan Sacks spoke.").text == "[PERSON_1] spoke."
        assert WithoutLord().anonymise("Lord Reed said so.").text == "Lord Reed said so."
