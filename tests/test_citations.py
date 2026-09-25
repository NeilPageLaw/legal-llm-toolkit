"""
Tests for citation parsing.
"""

import pytest

from legalkit.preprocess import CitationParser
from legalkit.preprocess.citations import Citation, deduplicate


@pytest.fixture
def parser():
    return CitationParser(jurisdiction="uk")


def only(parser, text):
    citations = parser.parse(text)
    assert len(citations) == 1, [c.raw for c in citations]
    return citations[0]


class TestUKCases:
    def test_neutral_citation(self, parser):
        citation = only(parser, "As held in Smith v Jones [2024] UKSC 15 at [42]")
        assert citation.year == 2024
        assert citation.court == "UKSC"
        assert citation.jurisdiction == "uk"
        assert citation.parties == "Smith v Jones"
        assert citation.paragraph == "42"
        assert citation.normalised == "[2024] UKSC 15"

    @pytest.mark.parametrize(
        "text, normalised, court",
        [
            ("[2023] EWCA Civ 123", "[2023] EWCA Civ 123", "EWCA Civ"),
            ("[2023] ewca  crim 7", "[2023] EWCA Crim 7", "EWCA Crim"),
            ("[1997] UKHL 14", "[1997] UKHL 14", "UKHL"),
            ("[2020] EWHC 1234 (Ch)", "[2020] EWHC 1234 (Ch)", "EWHC"),
            ("[2023] EWHC 99 (KB)", "[2023] EWHC 99 (KB)", "EWHC"),
            ("[2024] UKUT 123 (IAC)", "[2024] UKUT 123 (IAC)", "UKUT"),
            ("[2016] EWFC B67", "[2016] EWFC B67", "EWFC"),
            ("[2022] EAT 5", "[2022] EAT 5", "EAT"),
            ("[2020] CSIH 12", "[2020] CSIH 12", "CSIH"),
            ("[2021] NICA 3", "[2021] NICA 3", "NICA"),
            ("[2019] UKPC 1", "[2019] UKPC 1", "UKPC"),
        ],
    )
    def test_neutral_citation_courts(self, parser, text, normalised, court):
        citation = only(parser, text)
        assert citation.normalised == normalised
        assert citation.court == court

    @pytest.mark.parametrize(
        "text, normalised, reporter",
        [
            ("See [2024] 1 AC 123", "[2024] 1 AC 123", "AC"),
            ("[1932] AC 562", "[1932] AC 562", "AC"),
            ("[2023] KB 45", "[2023] KB 45", "KB"),
            ("[2008] 2 All ER (Comm) 1", "[2008] 2 All ER (Comm) 1", "All ER (Comm)"),
            ("[2010] 1 Lloyd's Rep 1", "[2010] 1 Lloyd's Rep 1", "Lloyd's Rep"),
            ("[2001] Lloyd’s Rep IR 123", "[2001] Lloyd's Rep IR 123", "Lloyd's Rep IR"),
            ("[1998] 1 WLR 896", "[1998] 1 WLR 896", "WLR"),
            ("[1990] 2 ac 605", "[1990] 2 AC 605", "AC"),
            ("(1854) 9 Exch 341", "(1854) 9 Exch 341", "Exch"),
            ("[1932] A.C. 562", "[1932] AC 562", "AC"),
            ("[1964] 1 Q.B. 5", "[1964] 1 QB 5", "QB"),
            ("[1997] 1 All E.R. 1", "[1997] 1 All ER 1", "All ER"),
            ("(1854) 9 Exch. 341", "(1854) 9 Exch 341", "Exch"),
            ("(1868) LR 3 HL 330", "(1868) LR 3 HL 330", "LR HL"),
        ],
    )
    def test_law_reports(self, parser, text, normalised, reporter):
        citation = only(parser, text)
        assert citation.normalised == normalised
        assert citation.reporter == reporter
        assert citation.citation_type == "case"

    def test_ehrr_is_tagged_echr(self, parser):
        citation = only(parser, "Soering v United Kingdom (1989) 11 EHRR 439")
        assert citation.jurisdiction == "echr"
        assert citation.parties == "Soering v United Kingdom"

    def test_multiple_citations(self, parser):
        text = """
        As established in [2020] UKSC 1 and confirmed in
        [2022] EWCA Civ 123, the principle applies.
        """
        assert [c.normalised for c in parser.parse(text)] == [
            "[2020] UKSC 1",
            "[2022] EWCA Civ 123",
        ]

    def test_following_citation_is_not_a_pinpoint(self, parser):
        first, second = parser.parse("See [2024] UKSC 1 and [2023] EWCA Civ 123.")
        assert first.paragraph is None
        assert second.paragraph is None

    def test_parallel_citation_is_not_a_pinpoint_and_shares_parties(self, parser):
        text = "Wood v Capita Insurance Services Ltd [2017] UKSC 24, [2017] AC 1173 at [10]"
        neutral, report = parser.parse(text)
        assert neutral.paragraph is None
        assert report.paragraph == "10"
        assert neutral.parties == report.parties == "Wood v Capita Insurance Services Ltd"

    @pytest.mark.parametrize(
        "text, paragraph",
        [
            ("[2019] UKSC 41 at [50]-[52]", "50-52"),
            ("[2015] UKSC 11, paras 82-87", "82-87"),
            ("[2015] UKSC 11 at para 5", "5"),
            ("[2015] UKSC 11, [7]", "7"),
            ("[2015] UKSC 11 and the court", None),
        ],
    )
    def test_paragraph_pinpoints(self, parser, text, paragraph):
        assert only(parser, text).paragraph == paragraph

    def test_repeated_citation_is_unique_by_default(self, parser):
        text = "[2020] UKSC 1 ... [2020]  UKSC 1 ... [2020] UKSC 1 at [3]"
        assert len(parser.parse(text)) == 1
        assert len(parser.parse(text, unique=False)) == 3


class TestCaseNames:
    @pytest.mark.parametrize(
        "text, parties",
        [
            ("As held in Smith v Jones [2024] UKSC 15", "Smith v Jones"),
            ("the House of Lords in Donoghue v Stevenson [1932] AC 562", "Donoghue v Stevenson"),
            ("R v Brown [1994] 1 AC 212", "R v Brown"),
            (
                "Caparo Industries plc v\n    Dickman [1990] 2 AC 605",
                "Caparo Industries plc v Dickman",
            ),
            (
                "in Robinson v Chief Constable of West Yorkshire Police [2018] UKSC 4",
                "Robinson v Chief Constable of West Yorkshire Police",
            ),
            (
                "R (Miller) v Secretary of State for Exiting the European Union [2017] UKSC 5",
                "R (Miller) v Secretary of State for Exiting the European Union",
            ),
            (
                "Secretary of State for the Home Department v AF (No 3) [2009] UKHL 28",
                "Secretary of State for the Home Department v AF (No 3)",
            ),
            ("Following Hunter v Canary Wharf Ltd [1997] UKHL 14", "Hunter v Canary Wharf Ltd"),
            ("Re Smith [2020] EWHC 1 (Ch)", "Re Smith"),
            ("Brown v. Board of Education, 347 U.S. 483 (1954)", "Brown v Board of Education"),
            ("Mata v. Avianca, Inc., 678 F. Supp. 3d 443 (S.D.N.Y. 2023)", "Mata v Avianca, Inc."),
            ("See [2024] 1 AC 123", None),
        ],
    )
    def test_party_extraction(self, parser, text, parties):
        assert only(parser, text).parties == parties

    def test_find_case_names_returns_offsets(self, parser):
        text = "The rule in Rylands v Fletcher (1868) LR 3 HL 330 applies."
        [(name, start, end)] = parser.find_case_names(text)
        assert name == "Rylands v Fletcher"
        assert text[start:end] == "Rylands v Fletcher"


class TestUKLegislation:
    @pytest.mark.parametrize(
        "text, normalised, raw",
        [
            (
                "Under section 1 of the Companies Act 2006, a company",
                "Companies Act 2006, s 1",
                "section 1 of the Companies Act 2006",
            ),
            (
                "section 2(1) of the Unfair Contract Terms Act 1977",
                "Unfair Contract Terms Act 1977, s 2(1)",
                "section 2(1) of the Unfair Contract Terms Act 1977",
            ),
            (
                "Companies Act 2006, s 994 permits",
                "Companies Act 2006, s 994",
                "Companies Act 2006, s 994",
            ),
            (
                "Schedule 1 to the Data Protection Act 2018",
                "Data Protection Act 2018, sch 1",
                "Schedule 1 to the Data Protection Act 2018",
            ),
            (
                "Jones and the Human Rights Act 1998 applies",
                "Human Rights Act 1998",
                "Human Rights Act 1998",
            ),
            (
                "the Representation of the People Act 1983",
                "Representation of the People Act 1983",
                "Representation of the People Act 1983",
            ),
            (
                "Legal Aid, Sentencing and Punishment of Offenders Act 2012",
                "Legal Aid, Sentencing and Punishment of Offenders Act 2012",
                "Legal Aid, Sentencing and Punishment of Offenders Act 2012",
            ),
        ],
    )
    def test_references(self, parser, text, normalised, raw):
        citation = only(parser, text)
        assert citation.citation_type == "legislation"
        assert citation.normalised == normalised
        assert citation.raw == raw

    def test_statutory_instrument_with_bracketed_title(self, parser):
        text = (
            "regulation 3 of the Consumer Contracts (Information, Cancellation and "
            "Additional Charges) Regulations 2013"
        )
        citation = only(parser, text)
        assert citation.provision == "reg 3"
        assert citation.year == 2013

    @pytest.mark.parametrize(
        "text",
        [
            "The claims 5 of the agreement were settled",
            "section 3 of the agreement between the parties shall act",
            "section 1 of the Act applies",
        ],
    )
    def test_no_false_positives(self, parser, text):
        assert parser.parse(text) == []

    def test_civil_procedure_rules(self, parser):
        text = "CPR r 3.4(2)(a), CPR 31.16, CPR Part 36 and CPR PD 57AD"
        assert [c.normalised for c in parser.parse(text)] == [
            "CPR r 3.4(2)(a)",
            "CPR r 31.16",
            "CPR Part 36",
            "CPR PD 57AD",
        ]


class TestUSCitations:
    def test_federal_reporter(self):
        citation = only(CitationParser("us"), "123 F.3d 456, 460 (9th Cir. 1997)")
        assert citation.volume == "123"
        assert citation.reporter == "F.3d"
        assert citation.court == "9th Cir."
        assert citation.year == 1997
        assert citation.jurisdiction == "us"

    def test_us_reports_year_only_parenthetical(self):
        citation = only(CitationParser("us"), "Roe v. Wade, 410 U.S. 113, 153 (1973)")
        assert citation.year == 1973
        assert citation.court is None
        assert citation.normalised == "410 U.S. 113"

    def test_reporter_spacing_is_normalised(self):
        citation = only(CitationParser("us"), "678 F. Supp.3d 443")
        assert citation.normalised == "678 F. Supp. 3d 443"

    def test_statutes(self):
        text = "42 U.S.C. § 1983 and 29 C.F.R. § 1630.2(g)"
        assert [c.normalised for c in CitationParser("us").parse(text)] == [
            "42 U.S.C. § 1983",
            "29 C.F.R. § 1630.2(g)",
        ]

    def test_single_letter_reporter_needs_full_stop(self):
        assert CitationParser("us").parse("the tenant paid 5 F 6 of rent") == []


class TestEUCitations:
    def test_case_number(self):
        citation = only(CitationParser("eu"), "In Case C-123/24, the Court held...")
        assert citation.jurisdiction == "eu"
        assert citation.year == 2024
        assert citation.court == "CJ"
        assert citation.normalised == "Case C-123/24"

    def test_joined_cases(self):
        text = "Joined Cases C-6/90 and C-9/90 Francovich"
        assert [c.normalised for c in CitationParser("eu").parse(text)] == [
            "Case C-6/90",
            "Case C-9/90",
        ]

    def test_general_court_and_ecli(self):
        text = "Case T-201/04 and ECLI:EU:C:2020:559"
        general, ecli = CitationParser("eu").parse(text)
        assert general.court == "GC"
        assert ecli.normalised == "ECLI:EU:C:2020:559"
        assert ecli.year == 2020

    def test_ecr(self):
        citation = only(CitationParser("eu"), "Costa v ENEL [1964] ECR 585")
        assert citation.parties == "Costa v ENEL"
        assert citation.jurisdiction == "eu"

    @pytest.mark.parametrize(
        "text, normalised, jurisdiction",
        [
            (
                "Article 6(1)(f) of Regulation (EU) 2016/679",
                "Art 6(1)(f) Regulation (EU) 2016/679",
                "eu",
            ),
            ("Article 267 TFEU", "Art 267 TFEU", "eu"),
            ("Article 47 of the Charter", "Art 47 Charter", "eu"),
            ("art 28(3)(a) UK GDPR", "Art 28(3)(a) UK GDPR", "uk"),
            ("Article 8 ECHR", "Art 8 ECHR", "echr"),
            ("Directive 95/46/EC", "Directive 95/46/EC", "eu"),
            ("Regulation (EC) No 1/2003", "Regulation (EC) No 1/2003", "eu"),
        ],
    )
    def test_legislation(self, text, normalised, jurisdiction):
        citation = only(CitationParser("eu"), text)
        assert citation.normalised == normalised
        assert citation.jurisdiction == jurisdiction
        assert citation.citation_type == "legislation"


class TestParserBehaviour:
    def test_all_formats_recognised_whatever_the_primary_jurisdiction(self):
        text = "[2020] UKSC 1, 347 U.S. 483 and Case C-6/90"
        for jurisdiction in ("uk", "us", "eu"):
            parsed = CitationParser(jurisdiction).parse(text)
            assert {c.jurisdiction for c in parsed} == {"uk", "us", "eu"}

    def test_jurisdictions_filter(self):
        text = "[2020] UKSC 1, 347 U.S. 483 and Case C-6/90"
        parsed = CitationParser("uk", jurisdictions=["us"]).parse(text)
        assert [c.jurisdiction for c in parsed] == ["us"]

    def test_unsupported_jurisdiction(self):
        with pytest.raises(ValueError, match="Unsupported jurisdiction"):
            CitationParser("xx")
        with pytest.raises(ValueError, match="Unsupported jurisdictions"):
            CitationParser("uk", jurisdictions=["xx"])

    def test_offsets_point_at_raw_text(self, parser):
        text = (
            "Donoghue v Stevenson [1932] AC 562; section 1 of the Companies Act 2006; "
            "Article 267 TFEU; 42 U.S.C. § 1983"
        )
        citations = parser.parse(text)
        assert len(citations) == 4
        for citation in citations:
            assert text[citation.start : citation.end] == citation.raw

    def test_extract_all_groups_by_type(self, parser):
        grouped = parser.extract_all("[2020] UKSC 1 and the Human Rights Act 1998")
        assert len(grouped["cases"]) == 1
        assert len(grouped["legislation"]) == 1
        assert grouped["other"] == []

    def test_to_dict(self, parser):
        record = only(parser, "[2020] UKSC 1").to_dict()
        assert record["type"] == "case"
        assert record["normalised"] == "[2020] UKSC 1"
        assert record["start"] == 0

    def test_deduplicate(self):
        a = Citation(raw="x", citation_type="case", jurisdiction="uk", normalised="[2020] UKSC 1")
        b = Citation(raw="y", citation_type="case", jurisdiction="uk", normalised="[2020] UKSC 1")
        assert deduplicate([a, b]) == [a]

    def test_large_input_is_fast(self, parser):
        import time

        text = ("Alpha of the Beta and Gamma " * 20000) + "[2020] UKSC 1"
        start = time.perf_counter()
        assert len(parser.parse(text)) == 1
        assert time.perf_counter() - start < 10


class TestReviewRegressions:
    @pytest.mark.parametrize(
        "text, parties",
        [
            ("A v B plc [2002] EWCA Civ 337", "A v B plc"),
            ("See A v B plc [2002] EWCA Civ 337", "A v B plc"),
            ("Von Hannover v Germany (2005) 40 EHRR 1", "Von Hannover v Germany"),
            ("De Keyser v Jones [1920] AC 508", "De Keyser v Jones"),
        ],
    )
    def test_short_parties_and_particles_are_kept(self, parser, text, parties):
        assert only(parser, text).parties == parties

    @pytest.mark.parametrize(
        "text, raw",
        [
            ("relied on the Human Rights\n    Act 1998", "Human Rights\n    Act 1998"),
            ("Human  Rights Act 1998, s 3 applies", "Human  Rights Act 1998, s 3"),
        ],
    )
    def test_act_offsets_survive_irregular_whitespace(self, parser, text, raw):
        citation = only(parser, text)
        assert citation.raw == raw
        assert text[citation.start : citation.end] == raw

    def test_case_name_offsets_skip_leading_words(self, parser):
        text = "Applying A v B plc [2002] EWCA Civ 337, the court held."
        [(name, start, end)] = parser.find_case_names(text)
        assert name == "A v B plc"
        assert text[start:end] == "A v B plc"
