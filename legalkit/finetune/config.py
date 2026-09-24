"""
Training configurations for legal LLM fine-tuning.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class FinetuneMethod(Enum):
    """Fine-tuning methods."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    PREFIX = "prefix"
    PROMPT = "prompt"


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


@dataclass
class LegalTrainingConfig:
    """
    Configuration for legal LLM fine-tuning.

    Provides sensible defaults for legal domain training
    with options for different fine-tuning methods.

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
    method: str = "qlora"  # full, lora, qlora, prefix, prompt

    # Task configuration
    task: str = "general"
    jurisdiction: str = "uk"

    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # LoRA/QLoRA specific
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    # QLoRA specific
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
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

    # Hugging Face Hub
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | None = None

    # Advanced
    seed: int = 42
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"

    # Legal-specific
    preserve_citations: bool = True
    anonymise_training_data: bool = False

    def __post_init__(self):
        """Validate and set defaults based on method/task."""
        # Set task-specific defaults
        self._apply_task_defaults()

        # Set method-specific defaults
        self._apply_method_defaults()

        # Set default LoRA target modules if not specified
        if self.lora_target_modules is None:
            self.lora_target_modules = self._default_target_modules()

    def _apply_task_defaults(self):
        """Apply task-specific configuration defaults."""
        task_configs = {
            LegalTask.CONTRACT_REVIEW.value: {
                "max_seq_length": 4096,  # Contracts can be long
                "num_epochs": 5,
            },
            LegalTask.CASE_ANALYSIS.value: {
                "max_seq_length": 4096,
                "num_epochs": 3,
            },
            LegalTask.LEGAL_QA.value: {
                "max_seq_length": 2048,
                "num_epochs": 3,
            },
            LegalTask.SUMMARISATION.value: {
                "max_seq_length": 4096,
                "num_epochs": 3,
            },
            LegalTask.CITATION_EXTRACTION.value: {
                "max_seq_length": 1024,
                "num_epochs": 5,
            },
        }

        if self.task in task_configs:
            for key, value in task_configs[self.task].items():
                # Only apply if not explicitly set (still default)
                if getattr(self, key) == getattr(LegalTrainingConfig(), key):
                    setattr(self, key, value)

    def _apply_method_defaults(self):
        """Apply method-specific configuration defaults."""
        if self.method == FinetuneMethod.QLORA.value:
            self.use_4bit = True
            self.gradient_checkpointing = True
            self.optim = "paged_adamw_32bit"
        elif self.method == FinetuneMethod.LORA.value:
            self.use_4bit = False
            self.gradient_checkpointing = True
        elif self.method == FinetuneMethod.FULL.value:
            self.use_4bit = False
            self.lora_r = 0
            self.lora_alpha = 0

    def _default_target_modules(self) -> list[str]:
        """Get default LoRA target modules for common architectures."""
        # These cover most Llama/Mistral-style models
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "base_model": self.base_model,
            "method": self.method,
            "task": self.task,
            "jurisdiction": self.jurisdiction,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "use_4bit": self.use_4bit,
            "output_dir": self.output_dir,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "LegalTrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        """Save config to JSON file."""
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LegalTrainingConfig":
        """Load config from JSON file."""
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def for_contract_review(cls, **kwargs) -> "LegalTrainingConfig":
        """Pre-configured for contract review task."""
        defaults = {
            "task": "contract_review",
            "max_seq_length": 4096,
            "num_epochs": 5,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_case_analysis(cls, **kwargs) -> "LegalTrainingConfig":
        """Pre-configured for case law analysis."""
        defaults = {
            "task": "case_analysis",
            "max_seq_length": 4096,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_legal_qa(cls, **kwargs) -> "LegalTrainingConfig":
        """Pre-configured for legal question answering."""
        defaults = {
            "task": "legal_qa",
            "max_seq_length": 2048,
        }
        defaults.update(kwargs)
        return cls(**defaults)
