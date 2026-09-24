"""
Fine-tuning module for legal LLMs.

Provides training configurations and wrappers for
LoRA, QLoRA, and full fine-tuning of language models.
"""

from legalkit.finetune.adapters import (
    create_lora_config,
    create_qlora_config,
    estimate_memory_usage,
    get_target_modules_for_model,
)
from legalkit.finetune.config import LegalTrainingConfig
from legalkit.finetune.trainer import LegalTrainer

__all__ = [
    "LegalTrainer",
    "LegalTrainingConfig",
    "create_lora_config",
    "create_qlora_config",
    "estimate_memory_usage",
    "get_target_modules_for_model",
]
