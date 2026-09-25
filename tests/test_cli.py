"""
Tests for the command-line interface.
"""

import json

import pytest

import legalkit
from legalkit.cli import main

JUDGMENT = (
    "1. Mr John Smith appeals. See Donoghue v Stevenson [1932] AC 562 and "
    "[2023] ewca civ 5.\n\n2. " + "The appeal is dismissed. " * 40
)


@pytest.fixture
def judgment_file(tmp_path):
    path = tmp_path / "judgment.txt"
    path.write_text(JUDGMENT)
    return path


def test_help_and_version(capsys):
    assert main([]) == 0
    assert "preprocess" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert legalkit.__version__ in capsys.readouterr().out


def test_info_links_to_the_repository(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "github.com/NeilPageLaw/legal-llm-toolkit" in out
    assert "YOUR_USERNAME" not in out


class TestPreprocess:
    def test_single_file(self, judgment_file, capsys):
        assert main(["preprocess", str(judgment_file), "--anonymise"]) == 0
        captured = capsys.readouterr()
        assert "[PERSON_1] appeals" in captured.out
        assert "[2023] EWCA Civ 5" in captured.out
        assert "Found 2 citations" in captured.err

    def test_chunks_are_written_as_jsonl(self, judgment_file, tmp_path):
        output = tmp_path / "chunks.jsonl"
        assert (
            main(
                [
                    "preprocess",
                    str(judgment_file),
                    "--chunk",
                    "--chunk-size",
                    "64",
                    "-o",
                    str(output),
                ]
            )
            == 0
        )
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(records) > 1
        assert records[0]["chunk_index"] == 0
        assert records[0]["source"] == "judgment.txt"
        # --chunk-size was previously ignored
        assert all(len(r["text"]) <= 64 * 4 + 100 for r in records)

    def test_directory(self, tmp_path, capsys):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text(JUDGMENT)
        (corpus / "b.txt").write_text(JUDGMENT)
        output = tmp_path / "out.jsonl"
        assert main(["preprocess", str(corpus), "--dedupe", "--anonymise", "-o", str(output)]) == 0
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(records) == 1
        assert "John Smith" not in records[0]["text"]
        assert "Removed 1 duplicates" in capsys.readouterr().err

    def test_directory_statistics_without_output(self, tmp_path, capsys):
        (tmp_path / "a.txt").write_text(JUDGMENT)
        assert main(["preprocess", str(tmp_path)]) == 0
        assert json.loads(capsys.readouterr().out)["num_samples"] == 1

    def test_missing_input(self, tmp_path, capsys):
        assert main(["preprocess", str(tmp_path / "missing.txt")]) == 1
        assert "No such file" in capsys.readouterr().err


class TestAnonymise:
    def test_writes_text_and_mapping(self, judgment_file, tmp_path, capsys):
        output = tmp_path / "out.txt"
        mapping = tmp_path / "mapping.json"
        assert (
            main(["anonymise", str(judgment_file), "-o", str(output), "--mapping", str(mapping)])
            == 0
        )
        assert "[PERSON_1]" in output.read_text()
        assert json.loads(mapping.read_text()) == {"Mr John Smith": "[PERSON_1]"}
        err = capsys.readouterr().err
        assert "person 1" in err
        assert "store it securely" in err

    def test_stdin_and_options(self, monkeypatch, capsys):
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("Mr Smith emailed a@b.com on 1 May 2024."))
        assert main(["anonymize", "-", "--entities", "email", "--keep-dates"]) == 0
        assert capsys.readouterr().out == "Mr Smith emailed [EMAIL_1] on 1 May 2024."

    def test_unknown_entity_type(self, judgment_file, capsys):
        assert main(["anonymise", str(judgment_file), "--entities", "pets"]) == 1
        assert "Unknown entity type" in capsys.readouterr().err


class TestCitations:
    def test_literal_text(self, capsys):
        assert main(["citations", "As held in Smith v Jones [2024] UKSC 15 at [42]"]) == 0
        out = capsys.readouterr().out
        assert "[case] [2024] UKSC 15" in out
        assert "Case: Smith v Jones" in out
        assert "Paragraph: 42" in out

    def test_long_text_is_not_mistaken_for_a_path(self, capsys):
        # Used to fail with "File name too long".
        assert main(["citations", "x " * 300 + "[2020] UKSC 1"]) == 0
        assert "[2020] UKSC 1" in capsys.readouterr().out

    def test_file_and_json(self, judgment_file, capsys):
        assert main(["citations", str(judgment_file), "--json"]) == 0
        records = json.loads(capsys.readouterr().out)
        assert [r["normalised"] for r in records] == ["[1932] AC 562", "[2023] EWCA Civ 5"]

    def test_no_citations(self, capsys):
        assert main(["citations", "Nothing to see here."]) == 0
        assert "No citations found." in capsys.readouterr().out


class TestTrain:
    def test_dry_run_merges_config_file_and_options(self, tmp_path, capsys):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"method": "lora", "num_epochs": 2, "lora_r": 8}))
        code = main(
            [
                "train",
                "data.jsonl",
                "--config",
                str(config),
                "--task",
                "contract_review",
                "--epochs",
                "4",
                "--dry-run",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        settings = json.loads(out[out.index("{") : out.rindex("}") + 1])
        assert settings["method"] == "lora"
        assert settings["lora_r"] == 8
        assert settings["num_epochs"] == 4  # the option overrides the file
        assert settings["max_seq_length"] == 4096  # task default
        assert "Estimated GPU memory" in out

    def test_unknown_config_setting(self, tmp_path, capsys):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"epochs": 2}))
        assert main(["train", "data.jsonl", "--config", str(config), "--dry-run"]) == 1
        assert "Unknown settings" in capsys.readouterr().err

    def test_invalid_choice(self, capsys):
        with pytest.raises(SystemExit):
            main(["train", "data.jsonl", "--method", "qlora2"])


def test_errors_show_traceback_only_when_verbose(capsys):
    assert main(["-v", "anonymise", "missing-file.txt"]) == 1
    assert "Traceback" in capsys.readouterr().err
    assert main(["anonymise", "missing-file.txt"]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_package_imports_without_ml_libraries():
    import subprocess
    import sys

    code = (
        "import sys, legalkit\n"
        "heavy = {'torch', 'transformers', 'peft', 'trl', 'datasets', 'bitsandbytes', 'spacy'}\n"
        "print(sorted(heavy & set(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_saved_config_can_be_reused_with_another_method(tmp_path, capsys):
    from legalkit.finetune import LegalTrainingConfig

    config = tmp_path / "training_config.json"
    LegalTrainingConfig(method="qlora").save(config)
    code = main(["train", "data.jsonl", "--config", str(config), "--method", "lora", "--dry-run"])
    assert code == 0, capsys.readouterr().err
    out = capsys.readouterr().out
    settings = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert settings["use_4bit"] is False
