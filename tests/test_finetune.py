"""
Tests for fine-tuning configuration and helpers (no ML libraries needed).
"""

import json

import pytest

from legalkit.finetune import (
    LegalTrainer,
    LegalTrainingConfig,
    estimate_memory_usage,
    get_target_modules_for_model,
)
from legalkit.finetune.adapters import estimate_parameters


class TestLegalTrainingConfig:
    def test_task_defaults_apply_when_unset(self):
        config = LegalTrainingConfig(task="contract_review")
        assert config.num_epochs == 5
        assert config.max_seq_length == 4096

    def test_explicit_values_are_never_overridden(self):
        # num_epochs=3 used to be silently replaced by the task default of 5.
        config = LegalTrainingConfig.for_contract_review(num_epochs=3, max_seq_length=2048)
        assert config.num_epochs == 3
        assert config.max_seq_length == 2048

    def test_method_defaults(self):
        qlora = LegalTrainingConfig(method="qlora")
        assert qlora.use_4bit is True
        assert qlora.optim == "paged_adamw_32bit"
        assert qlora.bnb_4bit_compute_dtype == "bfloat16"
        lora = LegalTrainingConfig(method="LoRA", bf16=False, fp16=True)
        assert lora.method == "lora"
        assert lora.use_4bit is False
        assert lora.optim == "adamw_torch"
        assert lora.bnb_4bit_compute_dtype == "float16"

    def test_explicit_method_settings_are_kept(self):
        config = LegalTrainingConfig(
            method="qlora", gradient_checkpointing=False, optim="adamw_torch"
        )
        assert config.gradient_checkpointing is False
        assert config.optim == "adamw_torch"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"method": "qlora2"}, "Unsupported method"),
            ({"method": "prefix"}, "Unsupported method"),
            ({"task": "mind_reading"}, "Unknown task"),
            ({"jurisdiction": "atlantis"}, "Unsupported jurisdiction"),
            ({"method": "lora", "use_4bit": True}, "qlora"),
            ({"fp16": True, "bf16": True}, "fp16 and bf16"),
            ({"batch_size": 0}, "batch_size"),
            ({"learning_rate": 0}, "learning_rate"),
            ({"lora_dropout": 1.5}, "lora_dropout"),
            ({"warmup_ratio": 1.0}, "warmup_ratio"),
        ],
    )
    def test_invalid_settings_raise(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            LegalTrainingConfig(**kwargs)

    def test_hub_token_is_hidden(self):
        config = LegalTrainingConfig(hub_token="hf_SECRET")
        assert "hf_SECRET" not in repr(config)
        assert "hub_token" not in config.to_dict()

    def test_hub_repositories_are_private_by_default(self):
        assert LegalTrainingConfig().hub_private_repo is True

    def test_save_load_round_trip_keeps_every_setting(self, tmp_path):
        config = LegalTrainingConfig(
            method="lora", task="legal_qa", warmup_ratio=0.1, lr_scheduler_type="linear"
        )
        path = tmp_path / "nested" / "config.json"
        config.save(path)
        saved = json.loads(path.read_text())
        assert saved["warmup_ratio"] == 0.1
        assert saved["lr_scheduler_type"] == "linear"
        assert LegalTrainingConfig.load(path) == config

    def test_from_dict_ignores_unknown_keys(self):
        config = LegalTrainingConfig.from_dict({"method": "lora", "colour": "blue"})
        assert config.method == "lora"

    def test_target_modules_follow_the_architecture(self):
        assert LegalTrainingConfig(base_model="gpt2").lora_target_modules == [
            "c_attn",
            "c_proj",
            "c_fc",
        ]
        assert "q_proj" in LegalTrainingConfig().lora_target_modules
        assert LegalTrainingConfig(base_model="acme/legal-lm").lora_target_modules == "all-linear"

    def test_presets(self):
        assert LegalTrainingConfig.for_case_analysis().max_seq_length == 4096
        assert LegalTrainingConfig.for_legal_qa().task == "legal_qa"


class TestAdapters:
    @pytest.mark.parametrize(
        "name, modules",
        [
            (
                "mistralai/Mistral-7B-v0.1",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            (
                "microsoft/Phi-3-mini-4k-instruct",
                ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            ),
            ("tiiuae/falcon-7b", ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]),
            ("unknown/model", "all-linear"),
        ],
    )
    def test_target_modules(self, name, modules):
        assert get_target_modules_for_model(name) == modules

    @pytest.mark.parametrize(
        "name, params",
        [
            ("mistralai/Mistral-7B-v0.1", 7e9),
            ("meta-llama/Llama-3.1-8B", 8e9),
            ("google/gemma-2-27b", 27e9),
            ("Qwen/Qwen2.5-0.5B-Instruct", 0.5e9),
            ("mistralai/Mixtral-8x7B-v0.1", 56e9),
            ("some-model-4bit", None),
            ("gpt2", None),
        ],
    )
    def test_estimate_parameters(self, name, params):
        assert estimate_parameters(name) == params

    def test_memory_estimate(self):
        qlora = estimate_memory_usage("mistral-7b", method="qlora", batch_size=4)
        full = estimate_memory_usage("mistral-7b", method="full", batch_size=4)
        assert qlora["total_estimated_gb"] < full["total_estimated_gb"]
        # Full fine-tuning with Adam needs ~16 bytes per parameter.
        assert full["model_memory_gb"] > 100
        assert full["recommended_gpu"] == "Multiple GPUs required"
        assert "error" in estimate_memory_usage("mystery-model")


class TestLegalTrainerWithoutModels:
    def test_save_before_training(self, tmp_path):
        trainer = LegalTrainer(LegalTrainingConfig())
        with pytest.raises(RuntimeError, match="not trained"):
            trainer.save(tmp_path)

    def test_anonymisation_refuses_opaque_datasets(self):
        trainer = LegalTrainer(LegalTrainingConfig(anonymise_training_data=True))
        with pytest.raises(ValueError, match="LegalDataset"):
            trainer._prepare_dataset(object())

    def test_empty_dataset(self, tmp_path):
        from legalkit.data import LegalDataset

        with pytest.raises(ValueError, match="empty"):
            LegalTrainer(LegalTrainingConfig())._prepare_dataset(LegalDataset())


class TestSavedConfigReuse:
    """Regressions from review: reusing a saved training_config.json."""

    def test_derived_settings_are_recorded(self):
        config = LegalTrainingConfig(task="contract_review", num_epochs=2)
        assert "num_epochs" not in config.derived_settings
        assert {"max_seq_length", "use_4bit", "optim", "lora_target_modules"} <= set(
            config.derived_settings
        )
        assert config.to_dict()["derived_settings"] == list(config.derived_settings)

    def test_reuse_with_another_method_model_or_task(self):
        saved = LegalTrainingConfig(method="qlora").to_dict()
        lora = LegalTrainingConfig.from_dict({**saved, "method": "lora"})
        assert lora.use_4bit is False
        assert lora.optim == "adamw_torch"
        gpt2 = LegalTrainingConfig.from_dict({**saved, "base_model": "gpt2"})
        assert gpt2.lora_target_modules == ["c_attn", "c_proj", "c_fc"]
        contracts = LegalTrainingConfig.from_dict({**saved, "task": "contract_review"})
        assert (contracts.num_epochs, contracts.max_seq_length) == (5, 4096)

    def test_explicit_values_survive_reuse(self):
        saved = LegalTrainingConfig(num_epochs=7, optim="adamw_torch").to_dict()
        reloaded = LegalTrainingConfig.from_dict({**saved, "task": "contract_review"})
        assert reloaded.num_epochs == 7
        assert reloaded.optim == "adamw_torch"


def test_full_fine_tuning_keeps_float32_weights():
    torch = pytest.importorskip("torch")
    trainer = LegalTrainer(LegalTrainingConfig(method="full", fp16=True, bf16=False))
    assert trainer._torch_dtype(torch) is torch.float32
    lora = LegalTrainer(LegalTrainingConfig(method="lora", fp16=True, bf16=False))
    assert lora._torch_dtype(torch) is torch.float16
