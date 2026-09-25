"""
Tests that the examples and sample data keep working.
"""

import importlib.util
from pathlib import Path

from legalkit.cli import main
from legalkit.eval import LegalBenchmark

EXAMPLES = Path(__file__).parent.parent / "examples"
SAMPLE_DATA = EXAMPLES / "sample_data"


def test_quickstart_runs(capsys):
    spec = importlib.util.spec_from_file_location("quickstart", EXAMPLES / "quickstart.py")
    quickstart = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quickstart)
    quickstart.main()
    out = capsys.readouterr().out
    assert "[2018] UKSC 4" in out
    assert "Not in the sources:  ['[2024] EWCA Civ 9999']" in out
    assert "Jane Doe" not in out.split("Anonymised excerpt:")[1].split("Entities found:")[0]


def test_sample_judgment_heading_is_anonymised(tmp_path, capsys):
    judgment = SAMPLE_DATA / "judgments" / "negligence_appeal.txt"
    assert main(["anonymise", str(judgment)]) == 0
    out = capsys.readouterr().out
    for name in ("ADAM CARTER", "Adam Carter", "DELTA LOGISTICS", "Eve Carter"):
        assert name not in out
    assert "Donoghue v Stevenson [1932] AC 562" in out


def test_sample_eval_data_is_valid():
    data = LegalBenchmark.load_test_data(SAMPLE_DATA / "eval")
    assert set(data) == {"contract_qa", "citation_accuracy"}
    suite = LegalBenchmark(show_progress=False).evaluate(generate_fn=lambda p: "", test_data=data)
    assert {r.task for r in suite.results} == set(data)


def test_sample_instructions_load():
    from legalkit.data import load_legal_corpus

    dataset = load_legal_corpus(SAMPLE_DATA / "instructions.jsonl")
    assert len(dataset) == 4
    assert all(sample.is_instruction for sample in dataset)


def test_readme_examples_produce_the_documented_output(capsys):
    from legalkit.preprocess import Anonymiser, CitationParser

    readme = (Path(__file__).parent.parent / "README.md").read_text()

    parser = CitationParser(jurisdiction="uk")
    text = "As held in Smith v Jones [2024] UKSC 15 at [42], applying section 1 of the Companies Act 2006"
    for citation in parser.parse(text):
        print(citation.normalised, citation.parties, citation.paragraph)
    for line in capsys.readouterr().out.splitlines():
        assert f"# {line}" in readme

    result = Anonymiser().anonymise(
        "Mr John Smith of 12 High Street, London SW1A 1AA relied on "
        "Caparo Industries plc v Dickman [1990] 2 AC 605."
    )
    assert f"# {result.text}" in readme
