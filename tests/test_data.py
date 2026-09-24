"""
Tests for the data module.
"""

import json

import pytest

from legalkit.data import LegalDataset, LegalSample, load_legal_corpus, to_instruction_format


class TestInstructionFormat:
    def test_alpaca_with_context(self):
        text = to_instruction_format("Name the parties.", "Acme Ltd and Beta LLP", context="...")
        assert "### Instruction:\nName the parties." in text
        assert "### Input:\n..." in text
        assert text.endswith("### Response:\nAcme Ltd and Beta LLP")

    def test_alpaca_without_context_omits_input_block(self):
        text = to_instruction_format("Define consideration.", "Something of value.")
        assert "### Input:" not in text
        assert text.endswith("### Response:\nSomething of value.")

    def test_prompt_only_ends_where_generation_starts(self):
        assert to_instruction_format("Define consideration.").endswith("### Response:\n")

    def test_chatml(self):
        text = to_instruction_format("Q", "A", context="C", template="chatml", system="S")
        assert text == (
            "<|im_start|>system\nS<|im_end|>\n"
            "<|im_start|>user\nQ\n\nC<|im_end|>\n"
            "<|im_start|>assistant\nA<|im_end|>"
        )

    def test_custom_template(self):
        text = to_instruction_format("Q", "A", template="Q: {instruction}\nA: {response}")
        assert text == "Q: Q\nA: A"

    def test_unknown_template(self):
        with pytest.raises(ValueError, match="Unknown template"):
            to_instruction_format("Q", "A", template="nonsense")


class TestLegalSample:
    def test_from_dict_accepts_alpaca_fields(self):
        sample = LegalSample.from_dict(
            {"instruction": "Summarise", "input": "The clause...", "output": "It limits..."}
        )
        assert sample.instruction == "Summarise"
        assert sample.text == "The clause..."
        assert sample.response == "It limits..."
        assert sample.is_instruction

    def test_from_dict_keeps_unknown_keys_as_metadata(self):
        sample = LegalSample.from_dict({"text": "x", "court": "UKSC", "metadata": {"a": 1}})
        assert sample.metadata == {"court": "UKSC", "a": 1}

    def test_from_dict_joins_paragraph_lists(self):
        sample = LegalSample.from_dict({"text": ["First fact.", "Second fact."]})
        assert sample.text == "First fact.\n\nSecond fact."

    def test_defaults_apply_only_when_missing(self):
        sample = LegalSample.from_dict(
            {"text": "x", "jurisdiction": "us"}, jurisdiction="uk", document_type="case"
        )
        assert sample.jurisdiction == "us"
        assert sample.document_type == "case"

    def test_round_trip(self):
        sample = LegalSample(
            text="t", instruction="i", response="r", document_type="contract", metadata={"k": 1}
        )
        assert LegalSample.from_dict(sample.to_dict()) == sample

    def test_raw_text_sample_trains_on_text(self):
        assert LegalSample(text="Raw judgment text").to_training_text() == "Raw judgment text"

    def test_to_messages(self):
        sample = LegalSample(text="Clause 1", instruction="Explain", response="It means...")
        assert sample.to_messages(system="S") == [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Explain\n\nClause 1"},
            {"role": "assistant", "content": "It means..."},
        ]

    def test_to_messages_requires_instruction(self):
        with pytest.raises(ValueError):
            LegalSample(text="x").to_messages()


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "contracts").mkdir()
    (tmp_path / "contracts" / "nda.txt").write_text("This Agreement is made between...")
    (tmp_path / "judgments").mkdir()
    (tmp_path / "judgments" / "smith.md").write_text("1. This appeal concerns...")
    records = [
        {"instruction": "Q1", "output": "A1"},
        {"instruction": "Q2", "output": "A2", "source": "manual"},
    ]
    (tmp_path / "qa.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (tmp_path / ".hidden.txt").write_text("secret")
    (tmp_path / "notes.pdf").write_text("ignored")
    return tmp_path


class TestLegalDatasetLoading:
    def test_from_directory(self, corpus_dir):
        dataset = LegalDataset.from_directory(corpus_dir, jurisdiction="uk")
        assert len(dataset) == 4
        by_source = {s.source: s for s in dataset}
        assert by_source["contracts/nda.txt"].document_type == "contract"
        assert by_source["judgments/smith.md"].document_type == "case"
        assert by_source["manual"].response == "A2"
        assert all(s.jurisdiction == "uk" for s in dataset)

    def test_from_directory_not_recursive(self, corpus_dir):
        assert len(LegalDataset.from_directory(corpus_dir, recursive=False)) == 2

    def test_from_directory_missing(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            LegalDataset.from_directory(tmp_path / "missing")

    def test_from_jsonl_reports_bad_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"text": "ok"}\n{not json}\n')
        with pytest.raises(ValueError, match=r"bad.jsonl:2"):
            LegalDataset.from_jsonl(path)

    def test_from_json(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"data": [{"text": "a"}, {"text": "b"}]}))
        assert [s.text for s in LegalDataset.from_json(path)] == ["a", "b"]

    def test_jsonl_round_trip(self, tmp_path):
        dataset = LegalDataset.from_texts(["one", "two"], document_type="case")
        path = dataset.to_jsonl(tmp_path / "out" / "data.jsonl")
        loaded = LegalDataset.from_jsonl(path)
        assert [s.text for s in loaded] == ["one", "two"]
        assert loaded[0].document_type == "case"

    def test_from_huggingface_rows(self):
        rows = [{"context": "c1", "label": 3}, {"context": "c2", "label": 4}]
        dataset = LegalDataset.from_huggingface(rows, limit=1, source="hub/id")
        assert len(dataset) == 1
        assert dataset[0].text == "c1"
        assert dataset[0].metadata == {"label": 3}
        assert dataset[0].source == "hub/id"

    def test_from_huggingface_text_field(self):
        dataset = LegalDataset.from_huggingface([{"opinion": "Held..."}], text_field="opinion")
        assert dataset[0].text == "Held..."

    def test_from_huggingface_missing_text_column(self):
        with pytest.raises(ValueError, match="text_field"):
            LegalDataset.from_huggingface([{"label": 1}])


class TestLoadLegalCorpus:
    def test_directory(self, corpus_dir):
        assert len(load_legal_corpus(corpus_dir)) == 4

    def test_text_file(self, corpus_dir):
        dataset = load_legal_corpus(corpus_dir / "contracts" / "nda.txt", document_type="contract")
        assert dataset[0].document_type == "contract"
        assert dataset[0].source == "nda.txt"

    def test_limit(self, corpus_dir):
        assert len(load_legal_corpus(corpus_dir / "qa.jsonl", limit=1)) == 1

    def test_missing_local_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_legal_corpus(tmp_path / "missing.jsonl")


class TestLegalDatasetOperations:
    def test_indexing_and_slicing(self):
        dataset = LegalDataset.from_texts(["a", "b", "c"])
        assert dataset[1].text == "b"
        assert isinstance(dataset[:2], LegalDataset)
        assert len(dataset[:2]) == 2

    def test_statistics(self):
        dataset = LegalDataset(
            [
                LegalSample(text="abcd", document_type="contract", jurisdiction="uk"),
                LegalSample(text="ab", instruction="q", response="r", document_type="case"),
            ]
        )
        stats = dataset.statistics()
        assert stats["num_samples"] == 2
        assert stats["num_instruction_samples"] == 1
        assert stats["total_chars"] == 8
        assert stats["min_chars"] == 4
        assert stats["document_types"] == {"contract": 1, "case": 1}
        assert stats["jurisdictions"] == {"uk": 1, "unknown": 1}

    def test_statistics_empty(self):
        assert LegalDataset().statistics()["avg_chars"] == 0

    def test_split_sizes_and_disjointness(self):
        dataset = LegalDataset.from_texts([str(i) for i in range(10)])
        train, val, test = dataset.split(train=0.8, val=0.1, test=0.1, seed=1)
        assert (len(train), len(val), len(test)) == (8, 1, 1)
        texts = [s.text for s in train] + [s.text for s in val] + [s.text for s in test]
        assert sorted(texts) == sorted(s.text for s in dataset)

    def test_split_is_reproducible(self):
        dataset = LegalDataset.from_texts([str(i) for i in range(20)])
        first = [s.text for s in dataset.split(seed=7)[0]]
        second = [s.text for s in dataset.split(seed=7)[0]]
        assert first == second

    def test_split_rejects_bad_fractions(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            LegalDataset.from_texts(["a"]).split(train=0.5, val=0.1, test=0.1)
        with pytest.raises(ValueError, match="negative"):
            LegalDataset.from_texts(["a"]).split(train=1.2, val=-0.2, test=0.0)

    def test_split_group_by_keeps_documents_together(self):
        samples = [
            LegalSample(text=f"chunk {i}", source=f"doc{i // 5}", metadata={"i": i})
            for i in range(20)
        ]
        train, val, test = LegalDataset(samples).split(0.5, 0.25, 0.25, group_by="source")
        for split_ in (train, val, test):
            for source in {s.source for s in split_}:
                assert sum(1 for s in samples if s.source == source) == sum(
                    1 for s in split_ if s.source == source
                )

    def test_deduplicate_ignores_case_and_whitespace(self):
        dataset = LegalDataset.from_texts(["The  Claimant", "the claimant", "Other"])
        assert [s.text for s in dataset.deduplicate()] == ["The  Claimant", "Other"]

    def test_filter_and_map(self):
        dataset = LegalDataset.from_texts(["short", "a much longer text"])
        assert len(dataset.filter(lambda s: len(s.text) > 5)) == 1
        upper = dataset.map(lambda s: LegalSample(text=s.text.upper()))
        assert upper[0].text == "SHORT"

    def test_to_huggingface(self):
        datasets = pytest.importorskip("datasets")
        dataset = LegalDataset(
            [LegalSample(text="Raw text"), LegalSample(instruction="Q", response="A")]
        )
        hf = dataset.to_huggingface()
        assert isinstance(hf, datasets.Dataset)
        assert hf[0]["text"] == "Raw text"
        assert hf[1]["text"].endswith("### Response:\nA")


class TestPreprocessAndChunk:
    def test_preprocess_anonymises_all_fields_consistently(self):
        dataset = LegalDataset(
            [
                LegalSample(
                    text="Mr John Smith emailed john@example.com.",
                    instruction="Who emailed?",
                    response="Mr John Smith did.",
                )
            ]
        )
        dataset.preprocess(anonymise=True)
        sample = dataset[0]
        assert "John Smith" not in sample.text
        assert "john@example.com" not in sample.text
        assert "John Smith" not in sample.response
        placeholder = sample.text.split(" emailed")[0]
        assert placeholder.startswith("[PERSON_")
        assert placeholder in sample.response
        assert sample.metadata["anonymised"] is True
        assert sample.jurisdiction == "uk"

    def test_preprocess_counts_citations(self):
        dataset = LegalDataset.from_texts(["See Donoghue v Stevenson [1932] AC 562."])
        dataset.preprocess()
        assert dataset[0].metadata["citation_count"] == 1

    def test_chunk_splits_documents_and_keeps_instructions(self):
        long_text = "\n\n".join(f"{i}. " + "The tenant shall pay rent. " * 20 for i in range(1, 6))
        dataset = LegalDataset(
            [
                LegalSample(text=long_text, source="lease"),
                LegalSample(instruction="Q", response="A"),
            ]
        )
        chunked = dataset.chunk(chunk_size=100, overlap=10)
        documents = [s for s in chunked if not s.is_instruction]
        assert len(documents) > 1
        assert all(s.source == "lease" for s in documents)
        assert [s.metadata["chunk_index"] for s in documents] == list(range(len(documents)))
        for s in documents:
            assert long_text[s.metadata["start_char"] : s.metadata["end_char"]] == s.text
        assert sum(1 for s in chunked if s.is_instruction) == 1
