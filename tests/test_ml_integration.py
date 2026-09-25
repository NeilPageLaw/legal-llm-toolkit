"""
End-to-end tests with a tiny randomly initialised model.

They need the training extras (torch, transformers, peft, trl, datasets)
and are skipped otherwise. The model and tokenizer are built locally, so no
downloads are needed. Run them with: pytest -m ml
"""

import json

import pytest

pytestmark = pytest.mark.ml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("datasets")
tokenizers = pytest.importorskip("tokenizers")

from legalkit.data import LegalDataset, LegalSample  # noqa: E402
from legalkit.eval import LegalBenchmark  # noqa: E402
from legalkit.finetune import LegalTrainer, LegalTrainingConfig  # noqa: E402

CORPUS = [
    "Donoghue v Stevenson [1932] AC 562 established the neighbour principle.",
    "The Supplier shall deliver the Goods on the Delivery Date.",
    "Either party may terminate this Agreement on 30 days' written notice.",
    "This Agreement is governed by the law of England and Wales.",
]

CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A 2-layer Llama with a byte-level BPE tokenizer, saved to disk."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny-llama")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        CORPUS * 20,
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.chat_template = CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(path)

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(hf_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


def cpu_config(model_path, output_dir, **overrides):
    settings = {
        "base_model": model_path,
        "method": "lora",
        "num_epochs": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 64,
        "lora_r": 4,
        "lora_alpha": 8,
        "bf16": False,
        "gradient_checkpointing": False,
        "logging_steps": 1,
        "save_steps": 1000,
        "output_dir": str(output_dir),
    }
    settings.update(overrides)
    return LegalTrainingConfig(**settings)


def test_lora_training_save_and_merge(tiny_model, tmp_path):
    trainer = LegalTrainer(cpu_config(tiny_model, tmp_path / "run"))
    result = trainer.train(LegalDataset.from_texts(CORPUS))
    assert result["train_loss"] > 0

    trainer.save(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    saved = json.loads((tmp_path / "adapter" / "training_config.json").read_text())
    assert saved["method"] == "lora"
    assert "hub_token" not in saved

    trainer.save(tmp_path / "merged", merge_adapter=True)
    merged = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
    assert merged.config.num_hidden_layers == 2


def test_lora_with_gradient_checkpointing_and_eval_set(tiny_model, tmp_path):
    config = cpu_config(tiny_model, tmp_path / "run", gradient_checkpointing=True, eval_steps=1)
    trainer = LegalTrainer(config)
    result = trainer.train(
        LegalDataset.from_texts(CORPUS), eval_dataset=LegalDataset.from_texts(CORPUS[:2])
    )
    assert result["train_loss"] > 0


def test_full_fine_tuning_from_jsonl_with_anonymisation(tiny_model, tmp_path):
    data = tmp_path / "train.jsonl"
    records = [
        {
            "instruction": "Who signed?",
            "input": "Mr John Smith signed on 1 May 2024.",
            "output": "Mr John Smith.",
        },
        {"instruction": "What law applies?", "input": CORPUS[3], "output": "English law."},
    ]
    data.write_text("\n".join(json.dumps(r) for r in records))
    config = cpu_config(tiny_model, tmp_path / "run", method="full", anonymise_training_data=True)
    trainer = LegalTrainer(config)
    prepared = trainer._prepare_dataset(data)
    assert "John Smith" not in prepared[0]["text"]
    assert trainer.train(data)["train_loss"] > 0


def test_chat_messages_template(tiny_model, tmp_path):
    dataset = LegalDataset(
        [LegalSample(text=text, instruction="Summarise.", response=text[:20]) for text in CORPUS]
    )
    trainer = LegalTrainer(cpu_config(tiny_model, tmp_path / "run", prompt_template="messages"))
    assert trainer.train(dataset)["train_loss"] > 0


def test_caller_dataset_is_not_modified(tiny_model, tmp_path):
    dataset = LegalDataset.from_texts(["Mr John Smith signed the lease."])
    trainer = LegalTrainer(cpu_config(tiny_model, tmp_path / "run", anonymise_training_data=True))
    trainer._prepare_dataset(dataset)
    assert dataset[0].text == "Mr John Smith signed the lease."


def test_benchmark_with_local_model(tiny_model):
    benchmark = LegalBenchmark(
        tasks=["contract_qa"], max_samples=1, max_new_tokens=5, show_progress=False
    )
    suite = benchmark.evaluate(model_path=tiny_model)
    result = suite.results[0]
    assert result.samples_evaluated == 1
    assert isinstance(result.per_sample_results[0]["response"], str)


def test_cli_train_and_evaluate(tiny_model, tmp_path, capsys):
    from legalkit.cli import main

    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps({"text": text}) for text in CORPUS))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "lora_r": 4,
                "lora_alpha": 8,
                "bf16": False,
                "gradient_checkpointing": False,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_seq_length": 64,
                "save_steps": 1000,
            }
        )
    )
    output = tmp_path / "model"
    code = main(
        [
            "train",
            str(data),
            "-m",
            tiny_model,
            "--method",
            "lora",
            "--epochs",
            "1",
            "--config",
            str(config),
            "-o",
            str(output),
            "--merge",
        ]
    )
    assert code == 0, capsys.readouterr().err
    assert (output / "training_config.json").exists()

    results = tmp_path / "results.json"
    code = main(
        [
            "evaluate",
            str(output),
            "--tasks",
            "contract_qa",
            "--max-samples",
            "1",
            "--max-new-tokens",
            "3",
            "-o",
            str(results),
        ]
    )
    assert code == 0, capsys.readouterr().err
    assert json.loads(results.read_text())["results"][0]["task"] == "contract_qa"


def test_benchmark_messages_template_uses_the_chat_template(tiny_model):
    benchmark = LegalBenchmark(
        tasks=["contract_qa"],
        max_samples=1,
        max_new_tokens=3,
        prompt_template="messages",
        show_progress=False,
    )
    suite = benchmark.evaluate(model_path=tiny_model)
    assert suite.results[0].samples_evaluated == 1
