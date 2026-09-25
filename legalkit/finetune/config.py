"""
Training configurations for legal LLM fine-tuning.
"""

import json
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any


def _as_json(value: Any) -> Any:
    """A value as it reads back from JSON (tuples become lists)."""
    return json.loads(json.dumps(value))


class FinetuneMethod(Enum):
    """Fine-tuning methods."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class LegalTask(Enum):
    """Pre-defined legal tasks."""

    GENERAL = "general"  # General legal language understanding
    CONTRACT_REVIEW = "contract_review"
    CASE_ANALYSIS = "case_analysis"
    LEGAL_QA = "legal_qa"
    DOCUMENT_DRAFTING = "document_drafting"
    CITATION_EXTRACTION = "citation_extraction"
    SUMMARISATION = "summarisation"
    CLAUSE_CLASSIFICATION = "clause_classification"


SUPPORTED_METHODS = tuple(method.value for method in FinetuneMethod)
SUPPORTED_TASKS = tuple(task.value for task in LegalTask)
SUPPORTED_JURISDICTIONS = ("uk", "us", "eu")

# The settings that derived defaults depend on.
DERIVATION_INPUTS = ("task", "method", "base_model", "bf16")

# Defaults applied when num_epochs or max_seq_length is not set explicitly.
GENERAL_DEFAULTS: dict[str, int] = {"num_epochs": 3, "max_seq_length": 2048}
TASK_DEFAULTS: dict[str, dict[str, int]] = {
    LegalTask.CONTRACT_REVIEW.value: {"max_seq_length": 4096, "num_epochs": 5},
    LegalTask.CASE_ANALYSIS.value: {"max_seq_length": 4096, "num_epochs": 3},
    LegalTask.LEGAL_QA.value: {"max_seq_length": 2048, "num_epochs": 3},
    LegalTask.SUMMARISATION.value: {"max_seq_length": 4096, "num_epochs": 3},
    LegalTask.CITATION_EXTRACTION.value: {"max_seq_length": 1024, "num_epochs": 5},
}


@dataclass
class LegalTrainingConfig:
    """
    Configuration for legal LLM fine-tuning.

    Provides sensible defaults for legal domain training with options for
    different fine-tuning methods. Settings left as None are filled in from
    the task and method: e.g. contract_review trains for 5 epochs on 4096
    tokens unless you set num_epochs or max_seq_length yourself.

    Example:
        >>> config = LegalTrainingConfig(
        ...     base_model="mistralai/Mistral-7B-v0.1",
        ...     method="qlora",
        ...     task="contract_review",
        ...     jurisdiction="uk"
        ... )
        >>> trainer = LegalTrainer(config)
    """

    # Model configuration
    base_model: str = "mistralai/Mistral-7B-v0.1"
    method: str = "qlora"  # full, lora, qlora

    # Task configuration
    task: str = "general"
    jurisdiction: str = "uk"

    # Training hyperparameters (None: task default)
    learning_rate: float = 2e-4
    num_epochs: int | None = None
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int | None = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # LoRA/QLoRA specific (target modules default to the model's architecture)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | str | None = None

    # QLoRA specific (None: 4-bit for qlora; compute dtype follows bf16/fp16)
    use_4bit: bool | None = None
    bnb_4bit_compute_dtype: str | None = None
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # Training settings
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3

    # Output configuration
    output_dir: str = "./legal-llm-output"
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_total_limit: int = 3

    # Dataset configuration
    train_dataset: str | None = None
    eval_dataset: str | None = None
    text_column: str = "text"
    prompt_template: str = "alpaca"  # alpaca, chatml, messages or a format string

    # Hugging Face Hub. Repositories are private by default: a model can
    # memorise and reveal its training data.
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | None = field(default=None, repr=False)
    hub_private_repo: bool = True

    # Advanced (optim None: paged_adamw_32bit for qlora, adamw_torch otherwise)
    seed: int = 42
    optim: str | None = None
    lr_scheduler_type: str = "cosine"
    report_to: str = "none"  # e.g. "tensorboard" or "wandb"
    trust_remote_code: bool = False  # only for model repositories you trust

    # Legal-specific
    preserve_citations: bool = True  # keep case names in citations when anonymising
    anonymise_training_data: bool = False

    # Names of the settings above that were filled in from the task, method
    # or model rather than set explicitly.
    derived_settings: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate settings and fill in task and method defaults."""
        self.method = self.method.lower()
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {self.method!r}. Choose one of: {', '.join(SUPPORTED_METHODS)}"
            )
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unknown task: {self.task!r}. Choose one of: {', '.join(SUPPORTED_TASKS)}"
            )
        self.jurisdiction = self.jurisdiction.lower()
        if self.jurisdiction not in SUPPORTED_JURISDICTIONS:
            raise ValueError(
                f"Unsupported jurisdiction: {self.jurisdiction!r}. "
                f"Choose one of: {', '.join(SUPPORTED_JURISDICTIONS)}"
            )

        if self.use_4bit and self.method != FinetuneMethod.QLORA.value:
            raise ValueError("4-bit loading is QLoRA: set method='qlora' instead of use_4bit=True")
        if self.fp16 and self.bf16:
            raise ValueError("Set at most one of fp16 and bf16")

        from legalkit.finetune.adapters import get_target_modules_for_model

        task_defaults = {**GENERAL_DEFAULTS, **TASK_DEFAULTS.get(self.task, {})}
        derived = {
            "num_epochs": lambda: task_defaults["num_epochs"],
            "max_seq_length": lambda: task_defaults["max_seq_length"],
            "use_4bit": lambda: self.method == FinetuneMethod.QLORA.value,
            "bnb_4bit_compute_dtype": lambda: "bfloat16" if self.bf16 else "float16",
            "optim": lambda: (
                "paged_adamw_32bit" if self.method == FinetuneMethod.QLORA.value else "adamw_torch"
            ),
            "lora_target_modules": lambda: get_target_modules_for_model(self.base_model),
        }
        filled = []
        for name, default in derived.items():
            if getattr(self, name) is None:
                setattr(self, name, default())
                filled.append(name)
        self.derived_settings = tuple(filled)

        for name in ("num_epochs", "batch_size", "gradient_accumulation_steps", "max_seq_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.method != FinetuneMethod.FULL.value:
            if self.lora_r < 1:
                raise ValueError("lora_r must be at least 1")
            if not 0 <= self.lora_dropout < 1:
                raise ValueError("lora_dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert config to dictionary.

        Every setting is included, so a saved config records exactly how a
        model was trained, with the names of settings derived from the task,
        method or model under "derived_settings". The Hub token is never
        included.
        """
        record = {
            f.name: getattr(self, f.name) for f in fields(self) if f.init and f.name != "hub_token"
        }
        record["derived_settings"] = list(self.derived_settings)
        record["derived_from"] = {name: getattr(self, name) for name in DERIVATION_INPUTS}
        return record

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "LegalTrainingConfig":
        """
        Create config from dictionary, ignoring unknown keys.

        Settings listed under "derived_settings" are derived again unless
        their value was edited, so a saved config reused with another method,
        task or model gets matching values.
        """
        return cls(**cls.explicit_settings(config_dict))

    @classmethod
    def explicit_settings(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        The settings in a config dictionary that were chosen rather than derived.

        A setting listed under "derived_settings" counts as chosen when its
        value differs from what the saved task, method and model give (recorded
        under "derived_from"), i.e. when someone edited it after saving. A
        config saved without "derived_from" cannot tell, so every listed
        setting is derived again.
        """
        names = {f.name for f in fields(cls) if f.init}
        settings = {k: v for k, v in config_dict.items() if k in names}
        derived = set(config_dict.get("derived_settings", [])) & set(settings)
        if not derived:
            return settings
        if "derived_from" not in config_dict:
            return {k: v for k, v in settings.items() if k not in derived}
        inputs = {k: settings[k] for k in DERIVATION_INPUTS if k in settings}
        inputs.update(config_dict["derived_from"])
        baseline = cls(**inputs)
        return {
            k: v
            for k, v in settings.items()
            if k not in derived or v != _as_json(getattr(baseline, k))
        }

    def save(self, path: str | Path):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LegalTrainingConfig":
        """Load config from JSON file."""
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def for_contract_review(cls, **kwargs) -> "LegalTrainingConfig":
        """Pre-configured for contract review task (4096 tokens, 5 epochs)."""
        return cls(**{"task": LegalTask.CONTRACT_REVIEW.value, **kwargs})

    @classmethod
    def for_case_analysis(cls, **kwargs) -> "LegalTrainingConfig":
        """Pre-configured for case law analysis (4096 tokens)."""
        return cls(**{"task": LegalTask.CASE_ANALYSIS.value, **kwargs})

    @classmethod
    def for_legal_qa(cls, **kwargs) -> "LegalTrainingConfig":
        """Pre-configured for legal question answering (2048 tokens)."""
        return cls(**{"task": LegalTask.LEGAL_QA.value, **kwargs})
