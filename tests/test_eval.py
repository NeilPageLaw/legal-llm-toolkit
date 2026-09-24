"""
Tests for evaluation metrics and benchmarks.
"""

import json

import pytest

from legalkit.eval import BenchmarkSuite, LegalBenchmark, LegalMetrics
from legalkit.eval.metrics import rouge_l, token_f1


@pytest.fixture
def metrics():
    return LegalMetrics(jurisdiction="uk")


class TestTextMetrics:
    def test_token_f1(self):
        assert token_f1("twelve months", "Twelve months.") == 1.0
        assert token_f1("", "") == 1.0
        assert token_f1("x", "") == 0.0
        assert token_f1("the term is 12 months", "12 months") == pytest.approx(4 / 7)

    def test_rouge_l(self):
        assert rouge_l("the cat sat", "the cat sat")["f1"] == 1.0
        # LCS "a c" of 3 and 2 tokens
        assert rouge_l("a b c", "a c")["f1"] == pytest.approx(0.8)
        assert rouge_l("", "a")["f1"] == 0.0


class TestCitationMetrics:
    def test_finds_all_uk_citation_forms(self, metrics):
        text = "Caparo [1990] 2 AC 605, [2022] EWCA Civ 123 and 347 U.S. 483."
        assert metrics.extract_citations(text) == [
            "[1990] 2 AC 605",
            "[2022] EWCA Civ 123",
            "347 U.S. 483",
        ]

    def test_precision_recall_with_normalisation(self, metrics):
        response = "See [1932] A.C. 562 and [2000]  1 WLR 5."
        result = metrics.evaluate_citations(response, ["[1932] AC 562", "[1990] 2 AC 605"])
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["missing_citations"] == ["[1990] 2 AC 605"]
        assert result["extra_citations"] == ["[2000] 1 WLR 5"]

    def test_no_expected_citations_reports_found_only(self, metrics):
        result = metrics.evaluate_citations("[2020] UKSC 1")
        assert result["citation_count"] == 1
        assert "precision" not in result

    def test_empty_expectation(self, metrics):
        assert metrics.evaluate_citations("No authorities.", [])["f1"] == 1.0
        assert metrics.evaluate_citations("[2020] UKSC 1", [])["precision"] == 0.0

    def test_legislation_counted_when_expected(self, metrics):
        response = "Section 3 of the Human Rights Act 1998 applies, see [2004] UKHL 30."
        result = metrics.evaluate_citations(response, ["Human Rights Act 1998, s 3"])
        assert result["recall"] == 1.0
        assert metrics.evaluate_citations(response, ["[2004] UKHL 30"])["precision"] == 1.0


class TestGrounding:
    def test_flags_citations_missing_from_sources(self, metrics):
        source = "In Donoghue v Stevenson [1932] AC 562 the House of Lords held..."
        response = "Donoghue v Stevenson [1932] AC 562 and Smith v Jones [2031] UKSC 99."
        result = metrics.evaluate_grounding(response, source)
        assert result["grounded"] == ["[1932] AC 562"]
        assert result["ungrounded"] == ["[2031] UKSC 99"]
        assert result["grounding_rate"] == 0.5

    def test_sources_can_be_citation_lists(self, metrics):
        result = metrics.evaluate_grounding("[1932] A.C. 562", ["[1932] AC 562"])
        assert result["grounding_rate"] == 1.0

    def test_nothing_cited_is_vacuously_grounded(self, metrics):
        assert metrics.evaluate_grounding("No authority.", "source")["grounding_rate"] == 1.0


class TestResponseMetrics:
    def test_evaluate_response(self, metrics):
        result = metrics.evaluate_response(
            response="The defendant owed a duty of care: Donoghue v Stevenson [1932] AC 562.",
            reference="A duty of care arises: Donoghue v Stevenson [1932] AC 562.",
            expected_citations=["[1932] AC 562"],
            sources=["[1932] AC 562"],
        )
        assert result["citation_metrics"]["f1"] == 1.0
        assert result["grounding_metrics"]["grounding_rate"] == 1.0
        assert "duty of care" in result["terminology_metrics"]["terms_used"]
        assert 0 < result["aggregate_score"] <= 1

    def test_no_citations_no_free_score(self, metrics):
        # A response with no citations used to score 1.0 on "format validity".
        result = metrics.evaluate_response("Banana.")
        assert result["aggregate_score"] == 0.0

    def test_hedging_counts_whole_words(self, metrics):
        assert metrics.evaluate_quality("The mayor was in dismay.")["hedging_score"] == 0
        assert metrics.evaluate_quality("It may apply. It may not.")["hedging_score"] == 2

    def test_similarity(self, metrics):
        result = metrics.evaluate_similarity("duty breach causation", "duty, breach and causation")
        assert result["word_recall"] == 0.75
        assert 0 < result["rouge_l_f1"] < 1


def echo_answers(prompt: str) -> str:
    """A fake model that answers the sample questions correctly."""
    answers = {
        "neighbour": "Donoghue v Stevenson [1932] AC 562.",
        "duration": "The agreement lasts 12 months.",
        "Classify": "Termination",
        "Summarise": "A manufacturer owes a duty of care to the ultimate consumer.",
        "Extract": "CASE: Donoghue v Stevenson\nCITATION: [1932] AC 562\nCOURT: House of Lords",
    }
    for key, answer in answers.items():
        if key in prompt:
            return answer
    return "I do not know."


class TestLegalBenchmark:
    def test_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            LegalBenchmark(tasks=["mind_reading"])

    def test_requires_a_model(self):
        with pytest.raises(ValueError, match="generate_fn"):
            LegalBenchmark(tasks=["contract_qa"]).evaluate()

    def test_all_tasks_run_on_builtin_samples(self):
        suite = LegalBenchmark(show_progress=False).evaluate(generate_fn=echo_answers)
        assert {r.task for r in suite.results} == set(LegalBenchmark.AVAILABLE_TASKS)
        # summary() used to crash on the legal_ner placeholder metric
        summary = suite.summary()
        assert "legal_ner" in summary
        for result in suite.results:
            assert all(isinstance(v, float) for v in result.metrics.values())

    def test_citation_accuracy(self):
        data = {"citation_accuracy": [{"prompt": "neighbour?", "citations": ["[1932] AC 562"]}]}
        suite = LegalBenchmark(tasks=["citation_accuracy"], show_progress=False).evaluate(
            generate_fn=echo_answers, test_data=data
        )
        assert suite.results[0].metrics["f1"] == 1.0

    def test_citation_grounding_with_context(self):
        data = {
            "citation_accuracy": [
                {
                    "prompt": "neighbour?",
                    "citations": ["[1932] AC 562"],
                    "context": "No cases here.",
                }
            ]
        }
        result = (
            LegalBenchmark(tasks=["citation_accuracy"], show_progress=False)
            .evaluate(generate_fn=echo_answers, test_data=data)
            .results[0]
        )
        assert result.metrics["grounding_rate"] == 0.0
        assert result.per_sample_results[0]["ungrounded_citations"] == ["[1932] AC 562"]

    def test_contract_qa(self):
        data = {
            "contract_qa": [
                {"context": "...", "question": "What is the duration?", "answers": ["12 months"]},
                {"context": "...", "question": "Who pays?", "answers": ["the Customer"]},
            ]
        }
        result = (
            LegalBenchmark(tasks=["contract_qa"], show_progress=False)
            .evaluate(generate_fn=echo_answers, test_data=data)
            .results[0]
        )
        assert result.metrics["exact_accuracy"] == 0.5
        assert 0 < result.metrics["token_f1"] < 1

    def test_classification_picks_the_first_option_mentioned(self):
        options = ["liability", "limitation of liability", "termination"]
        data = {
            "clause_classification": [
                {"text": "a", "label": "limitation of liability", "options": options},
                {"text": "b", "label": "liability", "options": options},
            ]
        }
        suite = LegalBenchmark(tasks=["clause_classification"], show_progress=False).evaluate(
            generate_fn=lambda prompt: "Limitation of liability, not termination.",
            test_data=data,
        )
        assert suite.results[0].metrics["accuracy"] == 0.5

    def test_ner(self):
        data = {
            "legal_ner": [
                {
                    "text": "In Donoghue v Stevenson [1932] AC 562 ...",
                    "entities": [
                        {"text": "Donoghue v Stevenson", "label": "CASE"},
                        {"text": "[1932] AC 562", "label": "CITATION"},
                        {"text": "Lord Atkin", "label": "JUDGE"},
                    ],
                }
            ]
        }
        metrics = (
            LegalBenchmark(tasks=["legal_ner"], show_progress=False)
            .evaluate(generate_fn=echo_answers, test_data=data)
            .results[0]
            .metrics
        )
        # COURT is not in this data's label set, so that line is ignored.
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == pytest.approx(2 / 3)

    def test_tasks_without_data_are_skipped_not_scored_zero(self):
        data = {"contract_qa": [{"question": "What is the duration?", "answers": ["12 months"]}]}
        suite = LegalBenchmark(
            tasks=["contract_qa", "legal_reasoning"], show_progress=False
        ).evaluate(generate_fn=echo_answers, test_data=data)
        assert [r.task for r in suite.results] == ["contract_qa"]
        assert suite.skipped == {"legal_reasoning": "no test data provided"}
        assert "Skipped" in suite.summary()

    def test_invalid_records_fail_before_generation(self):
        calls = []
        data = {"contract_qa": [{"question": "Q?"}]}
        with pytest.raises(ValueError, match="missing 'answers'"):
            LegalBenchmark(tasks=["contract_qa"]).evaluate(
                generate_fn=lambda p: calls.append(p) or "", test_data=data
            )
        assert calls == []

    def test_max_samples(self):
        suite = LegalBenchmark(tasks=["contract_qa"], max_samples=2, show_progress=False).evaluate(
            generate_fn=echo_answers
        )
        assert suite.results[0].samples_evaluated == 2

    def test_prompt_template(self):
        prompts = []
        LegalBenchmark(
            tasks=["contract_qa"], prompt_template="alpaca", show_progress=False
        ).evaluate(
            generate_fn=lambda p: prompts.append(p) or "",
            test_data={
                "contract_qa": [
                    {"context": "The term is 1 year.", "question": "Term?", "answers": ["1 year"]}
                ]
            },
        )
        assert "### Instruction:\nTerm?" in prompts[0]
        assert "### Input:\nThe term is 1 year." in prompts[0]

    def test_load_test_data_from_directory(self, tmp_path):
        (tmp_path / "contract_qa.jsonl").write_text(
            json.dumps({"question": "Q?", "answers": ["A"]}) + "\n"
        )
        (tmp_path / "notes.json").write_text("[]")
        data = LegalBenchmark.load_test_data(tmp_path)
        assert data == {"contract_qa": [{"question": "Q?", "answers": ["A"]}]}
        suite = LegalBenchmark(tasks=["contract_qa"], show_progress=False).evaluate(
            generate_fn=lambda p: "A", test_data=tmp_path
        )
        assert suite.results[0].metrics["exact_accuracy"] == 1.0

    def test_load_test_data_errors(self, tmp_path):
        with pytest.raises(ValueError, match="No test data"):
            LegalBenchmark.load_test_data(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"mind_reading": []}))
        with pytest.raises(ValueError, match="unknown tasks"):
            LegalBenchmark.load_test_data(bad)


class TestBenchmarkSuite:
    def test_save_includes_samples_and_skips(self, tmp_path):
        suite = LegalBenchmark(tasks=["contract_qa"], show_progress=False).evaluate(
            generate_fn=echo_answers
        )
        suite.skipped["legal_ner"] = "no samples"
        path = suite.save(tmp_path / "out" / "results.json")
        saved = json.loads(path.read_text())
        assert saved["results"][0]["per_sample_results"]
        assert saved["skipped"] == {"legal_ner": "no samples"}
        without = json.loads(suite.save(tmp_path / "r.json", include_samples=False).read_text())
        assert "per_sample_results" not in without["results"][0]

    def test_empty_suite_summary(self):
        assert "RESULTS" in BenchmarkSuite().summary()
