"""
Legal LLM Trainer - Main training orchestration.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

from legalkit.finetune.config import LegalTrainingConfig, FinetuneMethod
from legalkit.data.dataset import LegalDataset


logger = logging.getLogger(__name__)


class LegalTrainer:
    """
    Main trainer class for legal LLM fine-tuning.
    
    Wraps Hugging Face Transformers and PEFT for easy
    fine-tuning with legal-domain optimizations.
    
    Example:
        >>> config = LegalTrainingConfig(
        ...     base_model="mistralai/Mistral-7B-v0.1",
        ...     method="qlora",
        ...     task="contract_review"
        ... )
        >>> trainer = LegalTrainer(config)
        >>> trainer.train("./contracts_dataset/")
        >>> trainer.save("./legal-mistral")
    """
    
    def __init__(self, config: LegalTrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._is_trained = False
        
    def train(
        self,
        dataset: Union[str, Path, LegalDataset],
        eval_dataset: Optional[Union[str, Path, LegalDataset]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model on legal data.
        
        Args:
            dataset: Training dataset (path or LegalDataset)
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Dictionary with training metrics
        """
        # Import here to avoid loading heavy dependencies on import
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )
            from peft import prepare_model_for_kbit_training, get_peft_model
            from trl import SFTTrainer
        except ImportError as e:
            raise ImportError(
                f"Missing required dependency: {e}. "
                "Install with: pip install legal-llm-toolkit[train]"
            )
        
        logger.info(f"Loading base model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with quantization if QLoRA
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if self.config.method == FinetuneMethod.QLORA.value and self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs
        )
        
        # Prepare for training
        if self.config.method in [FinetuneMethod.LORA.value, FinetuneMethod.QLORA.value]:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
            
            # Apply LoRA
            from legalkit.finetune.adapters import create_lora_config
            lora_config = create_lora_config(self.config)
            self.model = get_peft_model(self.model, lora_config)
            
            trainable_params, total_params = self._count_parameters()
            logger.info(
                f"Trainable params: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )
        
        # Prepare dataset
        train_data = self._prepare_dataset(dataset)
        eval_data = self._prepare_dataset(eval_dataset) if eval_dataset else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_data else None,
            evaluation_strategy="steps" if eval_data else "no",
            save_total_limit=self.config.save_total_limit,
            push_to_hub=self.config.push_to_hub,
            hub_model_id=self.config.hub_model_id,
            hub_token=self.config.hub_token,
            seed=self.config.seed,
            report_to="none",  # Can be changed to wandb, tensorboard, etc.
        )
        
        # Use SFTTrainer for instruction tuning
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field=self.config.text_column,
        )
        
        logger.info("Starting training...")
        train_result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        self._is_trained = True
        
        return {
            "train_loss": train_result.training_loss,
            "metrics": train_result.metrics,
        }
    
    def save(
        self,
        output_path: Union[str, Path],
        merge_adapter: bool = False,
    ):
        """
        Save the trained model.
        
        Args:
            output_path: Path to save model
            merge_adapter: Whether to merge LoRA weights into base model
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
            
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if merge_adapter and self.config.method in ["lora", "qlora"]:
            logger.info("Merging adapter weights...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_path)
        else:
            self.model.save_pretrained(output_path)
            
        self.tokenizer.save_pretrained(output_path)
        self.config.save(output_path / "training_config.json")
        
        logger.info(f"Model saved to {output_path}")
    
    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = True,
    ):
        """
        Push trained model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID (username/model-name)
            token: HF API token
            private: Whether repository should be private
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
            
        self.model.push_to_hub(
            repo_id,
            token=token,
            private=private,
        )
        self.tokenizer.push_to_hub(
            repo_id,
            token=token,
            private=private,
        )
        
        logger.info(f"Model pushed to hub: {repo_id}")
    
    def _prepare_dataset(
        self,
        dataset: Union[str, Path, LegalDataset, None]
    ):
        """Prepare dataset for training."""
        if dataset is None:
            return None
            
        if isinstance(dataset, LegalDataset):
            return dataset.to_huggingface()
        elif isinstance(dataset, (str, Path)):
            path = Path(dataset)
            if path.is_dir():
                ds = LegalDataset.from_directory(path)
            elif path.suffix == ".jsonl":
                ds = LegalDataset.from_jsonl(path)
            else:
                raise ValueError(f"Unsupported dataset format: {path}")
            
            # Preprocess if configured
            if self.config.anonymise_training_data:
                ds.preprocess(
                    anonymise=True,
                    jurisdiction=self.config.jurisdiction
                )
                
            return ds.to_huggingface()
        else:
            # Assume it's already a HF dataset
            return dataset
    
    def _count_parameters(self):
        """Count trainable and total parameters."""
        trainable = 0
        total = 0
        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        return trainable, total
