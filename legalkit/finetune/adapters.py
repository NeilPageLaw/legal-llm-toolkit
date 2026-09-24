"""
Adapter configurations for LoRA and QLoRA fine-tuning.
"""

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from legalkit.finetune.config import LegalTrainingConfig

LLAMA_STYLE_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def create_lora_config(config: "LegalTrainingConfig"):
    """
    Create LoRA configuration from training config.

    Args:
        config: LegalTrainingConfig instance

    Returns:
        LoraConfig for PEFT
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as e:
        raise ImportError(
            "Install the training extras: pip install 'legal-llm-toolkit[train]'"
        ) from e

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def create_qlora_config(config: "LegalTrainingConfig", compute_dtype: str | None = None):
    """
    Create QLoRA (4-bit) configuration.

    Args:
        config: LegalTrainingConfig instance
        compute_dtype: Computation dtype (default: config.bnb_4bit_compute_dtype)

    Returns:
        Tuple of (LoraConfig, BitsAndBytesConfig)
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise ImportError("Install the QLoRA extras: pip install 'legal-llm-toolkit[qlora]'") from e

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype or config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )
    return create_lora_config(config), bnb_config


def get_target_modules_for_model(model_name: str) -> list[str] | str:
    """
    Get recommended LoRA target modules for a model.

    Args:
        model_name: Model name or path

    Returns:
        List of module names to target, or "all-linear" (every linear layer
        except the output head) for architectures not listed here
    """
    model_lower = model_name.lower()

    # Llama, Mistral, Qwen, Gemma and similar architectures
    if any(
        x in model_lower for x in ["llama", "mistral", "mixtral", "qwen", "gemma", "yi-", "smollm"]
    ):
        return list(LLAMA_STYLE_MODULES)

    # Phi-3 and later fuse the attention and MLP projections
    if re.search(r"phi-?[34]", model_lower):
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

    # Falcon
    if "falcon" in model_lower:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    # MPT
    if "mpt" in model_lower:
        return ["Wqkv", "out_proj", "up_proj", "down_proj"]

    # GPT-NeoX / Pythia
    if any(x in model_lower for x in ["neox", "pythia"]):
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    # GPT-2
    if "gpt2" in model_lower:
        return ["c_attn", "c_proj", "c_fc"]

    return "all-linear"


def estimate_parameters(model_name: str) -> int | None:
    """
    Estimate a model's parameter count from its name, e.g. "Llama-3.1-8B" or
    "Mixtral-8x7B". Mixture-of-experts models are counted as experts x size,
    an over-estimate.

    Returns:
        Parameter count, or None if the name gives no size
    """
    match = re.search(r"(?<![\d.])(?:(\d+)x)?(\d+(?:\.\d+)?)b(?![a-z])", model_name.lower())
    if match is None:
        return None
    experts = int(match.group(1) or 1)
    return int(experts * float(match.group(2)) * 1_000_000_000)


def estimate_memory_usage(
    model_name: str,
    method: str = "qlora",
    batch_size: int = 4,
    max_seq_length: int = 2048,
) -> dict[str, Any]:
    """
    Estimate VRAM usage for training.

    A rough guide for choosing hardware, not a guarantee: real usage depends
    on the architecture, optimizer and sequence packing.

    Args:
        model_name: Model name or path
        method: Fine-tuning method
        batch_size: Training batch size
        max_seq_length: Maximum sequence length

    Returns:
        Dictionary with memory estimates
    """
    params = estimate_parameters(model_name)
    if params is None:
        return {"error": "Could not determine model size from its name"}

    # Memory estimates (very rough)
    if method == "qlora":
        # 4-bit quantization: ~0.5 bytes per param
        model_memory_gb = (params * 0.5) / (1024**3)
        # Plus gradients and optimizer states for LoRA params only
        training_overhead_gb = 2.0
    elif method == "lora":
        # 16-bit weights: ~2 bytes per param
        model_memory_gb = (params * 2) / (1024**3)
        training_overhead_gb = 4.0
    else:  # full
        # 16-bit weights and gradients plus 32-bit Adam states: ~16 bytes per param
        model_memory_gb = (params * 16) / (1024**3)
        training_overhead_gb = 0.0

    # Activation memory (rough estimate)
    activation_memory_gb = (batch_size * max_seq_length * 4096 * 4) / (1024**3)

    total_gb = model_memory_gb + training_overhead_gb + activation_memory_gb

    return {
        "model_memory_gb": round(model_memory_gb, 1),
        "training_overhead_gb": round(training_overhead_gb, 1),
        "activation_memory_gb": round(activation_memory_gb, 1),
        "total_estimated_gb": round(total_gb, 1),
        "recommended_gpu": _recommend_gpu(total_gb),
    }


def _recommend_gpu(memory_gb: float) -> str:
    """Recommend GPU based on memory requirements."""
    if memory_gb <= 8:
        return "RTX 3070/4070 (8GB) or better"
    elif memory_gb <= 12:
        return "RTX 3080/4080 (12GB) or better"
    elif memory_gb <= 16:
        return "RTX 4080 Super (16GB) or V100"
    elif memory_gb <= 24:
        return "RTX 3090/4090 (24GB) or A10"
    elif memory_gb <= 48:
        return "A40/A6000 (48GB)"
    elif memory_gb <= 80:
        return "A100/H100 (80GB)"
    else:
        return "Multiple GPUs required"
