"""
Fine-tuning module for legal LLMs.

Provides training configurations and wrappers for
LoRA, QLoRA, and full fine-tuning of language models.
"""

from legalkit.finetune.trainer import LegalTrainer
from legalkit.finetune.config import LegalTrainingConfig
from legalkit.finetune.adapters import (
    create_lora_config,
    create_qlora_config,
)

__all__ = [
    "LegalTrainer",
    "LegalTrainingConfig",
    "create_lora_config",
    "create_qlora_config",
]
