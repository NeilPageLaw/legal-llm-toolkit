"""
Adapter configurations for LoRA and QLoRA fine-tuning.
"""

from typing import List, Optional
from legalkit.finetune.config import LegalTrainingConfig


def create_lora_config(config: LegalTrainingConfig):
    """
    Create LoRA configuration from training config.
    
    Args:
        config: LegalTrainingConfig instance
        
    Returns:
        LoraConfig for PEFT
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        raise ImportError("Install peft: pip install peft")
    
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def create_qlora_config(
    config: LegalTrainingConfig,
    compute_dtype: str = "float16"
):
    """
    Create QLoRA (4-bit) configuration.
    
    Args:
        config: LegalTrainingConfig instance
        compute_dtype: Computation dtype
        
    Returns:
        Tuple of (LoraConfig, BitsAndBytesConfig)
    """
    try:
        from peft import LoraConfig, TaskType
        from transformers import BitsAndBytesConfig
        import torch
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )
    
    return lora_config, bnb_config


def get_target_modules_for_model(model_name: str) -> List[str]:
    """
    Get recommended LoRA target modules for a model.
    
    Args:
        model_name: Model name or path
        
    Returns:
        List of module names to target
    """
    model_lower = model_name.lower()
    
    # Llama, Mistral, and similar architectures
    if any(x in model_lower for x in ["llama", "mistral", "mixtral", "qwen"]):
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # Falcon
    if "falcon" in model_lower:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    # MPT
    if "mpt" in model_lower:
        return ["Wqkv", "out_proj", "up_proj", "down_proj"]
    
    # GPT-NeoX / Pythia
    if any(x in model_lower for x in ["neox", "pythia"]):
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    # Default - common attention modules
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def estimate_memory_usage(
    model_name: str,
    method: str = "qlora",
    batch_size: int = 4,
    max_seq_length: int = 2048,
) -> dict:
    """
    Estimate VRAM usage for training.
    
    Args:
        model_name: Model name or path
        method: Fine-tuning method
        batch_size: Training batch size
        max_seq_length: Maximum sequence length
        
    Returns:
        Dictionary with memory estimates
    """
    # Rough parameter counts for common models
    model_params = {
        "7b": 7_000_000_000,
        "13b": 13_000_000_000,
        "34b": 34_000_000_000,
        "70b": 70_000_000_000,
    }
    
    # Try to infer model size
    model_lower = model_name.lower()
    params = None
    for size, count in model_params.items():
        if size in model_lower:
            params = count
            break
    
    if params is None:
        return {"error": "Could not determine model size"}
    
    # Memory estimates (very rough)
    if method == "qlora":
        # 4-bit quantization: ~0.5 bytes per param
        model_memory_gb = (params * 0.5) / (1024**3)
        # Plus gradients and optimizer states for LoRA params only
        lora_overhead_gb = 2.0  # Rough estimate
    elif method == "lora":
        # FP16: ~2 bytes per param
        model_memory_gb = (params * 2) / (1024**3)
        lora_overhead_gb = 4.0
    else:  # full
        # FP16 with gradients: ~4 bytes per param
        model_memory_gb = (params * 4) / (1024**3)
        lora_overhead_gb = 0
    
    # Activation memory (rough estimate)
    activation_memory_gb = (batch_size * max_seq_length * 4096 * 4) / (1024**3)
    
    total_gb = model_memory_gb + lora_overhead_gb + activation_memory_gb
    
    return {
        "model_memory_gb": round(model_memory_gb, 1),
        "training_overhead_gb": round(lora_overhead_gb, 1),
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
        return "A100 (80GB)"
    else:
        return "Multiple GPUs required"
