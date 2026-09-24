"""
Legal LLM Trainer - Main training orchestration.
"""

import logging
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from legalkit.data.dataset import LegalDataset
from legalkit.data.loaders import load_legal_corpus
from legalkit.finetune.config import FinetuneMethod, LegalTrainingConfig

logger = logging.getLogger(__name__)

INSTALL_HINT = "Install the training extras: pip install 'legal-llm-toolkit[train]'"


class LegalTrainer:
    """
    Main trainer class for legal LLM fine-tuning.

    Wraps Hugging Face Transformers, PEFT and TRL for easy fine-tuning with
    legal-domain defaults. Supports full fine-tuning, LoRA and QLoRA (4-bit,
    needs a CUDA GPU and ``pip install 'legal-llm-toolkit[qlora]'``).

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
        dataset: str | Path | LegalDataset | Any,
        eval_dataset: str | Path | LegalDataset | Any | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """
        Train the model on legal data.

        Args:
            dataset: Training data: a LegalDataset, a path (directory,
                .jsonl, .json or .txt), a Hugging Face dataset id, or a
                ``datasets.Dataset`` with a text (or messages) column
            eval_dataset: Optional evaluation data, in the same forms
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Dictionary with training metrics
        """
        # Import here to avoid loading heavy dependencies on import
        try:
            import torch
            from peft import get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(f"Missing training dependency ({e.name}). {INSTALL_HINT}") from e

        from legalkit._compat import dtype_kwargs
        from legalkit.finetune.adapters import create_lora_config

        config = self.config
        # Prepare data first, so data problems surface before a large download.
        train_data = self._prepare_dataset(dataset)
        eval_data = self._prepare_dataset(eval_dataset) if eval_dataset is not None else None

        logger.info(f"Loading base model: {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code=config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            **dtype_kwargs(self._torch_dtype(torch)),
        }
        if config.method == FinetuneMethod.QLORA.value:
            _, model_kwargs["quantization_config"] = _qlora_quantization(config)
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)

        if config.method == FinetuneMethod.QLORA.value:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=config.gradient_checkpointing
            )
        elif config.method == FinetuneMethod.LORA.value and config.gradient_checkpointing:
            self.model.enable_input_require_grads()

        if config.method in (FinetuneMethod.LORA.value, FinetuneMethod.QLORA.value):
            self.model = get_peft_model(self.model, create_lora_config(config))
            trainable_params, total_params = self._count_parameters()
            logger.info(
                f"Trainable params: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

        self.trainer = self._build_trainer(train_data, eval_data)

        logger.info("Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self._is_trained = True

        return {
            "train_loss": train_result.training_loss,
            "metrics": train_result.metrics,
        }

    def save(self, output_path: str | Path, merge_adapter: bool = False):
        """
        Save the trained model.

        Args:
            output_path: Path to save model
            merge_adapter: Merge LoRA weights into the base model, giving a
                standalone model. For QLoRA the base model is reloaded in
                16-bit to merge, as merging into 4-bit weights loses precision;
                this needs enough memory for the full 16-bit model.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if merge_adapter and self.config.method == FinetuneMethod.LORA.value:
            logger.info("Merging adapter weights...")
            self.model.merge_and_unload().save_pretrained(output_path)
        elif merge_adapter and self.config.method == FinetuneMethod.QLORA.value:
            logger.info("Reloading the base model in 16-bit to merge the adapter...")
            self._merge_into_16bit_base().save_pretrained(output_path)
        else:
            self.model.save_pretrained(output_path)

        self.tokenizer.save_pretrained(output_path)
        self.config.save(output_path / "training_config.json")

        logger.info(f"Model saved to {output_path}")

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = True,
    ):
        """
        Push trained model to Hugging Face Hub.

        A model can memorise and reproduce its training data. Keep the
        repository private unless the training data is public.

        Args:
            repo_id: Repository ID (username/model-name)
            token: HF API token
            private: Whether repository should be private
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        self.model.push_to_hub(repo_id, token=token, private=private)
        self.tokenizer.push_to_hub(repo_id, token=token, private=private)

        logger.info(f"Model pushed to hub: {repo_id}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_dataset(self, dataset):
        """Prepare dataset for training."""
        if dataset is None:
            return None

        if isinstance(dataset, LegalDataset):
            # Copy, so anonymisation does not modify the caller's dataset.
            legal_dataset = LegalDataset(replace(s, metadata=dict(s.metadata)) for s in dataset)
        elif isinstance(dataset, (str, Path)):
            legal_dataset = load_legal_corpus(dataset)
        else:
            if self.config.anonymise_training_data:
                raise ValueError(
                    "anonymise_training_data needs a LegalDataset or a path. Convert "
                    "a Hugging Face dataset with LegalDataset.from_huggingface() first."
                )
            return dataset  # assume a ready-made Hugging Face dataset

        if not len(legal_dataset):
            raise ValueError("The training dataset is empty")
        if self.config.anonymise_training_data:
            legal_dataset.preprocess(
                anonymise=True,
                jurisdiction=self.config.jurisdiction,
                preserve_case_names=self.config.preserve_citations,
            )
        return legal_dataset.to_huggingface(
            template=self.config.prompt_template, text_column=self.config.text_column
        )

    def _build_trainer(self, train_data, eval_data):
        """Create an SFTTrainer, adapting to the installed transformers/TRL versions."""
        try:
            from trl import SFTConfig, SFTTrainer
        except ImportError as e:
            raise ImportError(f"Missing training dependency (trl). {INSTALL_HINT}") from e

        from legalkit._compat import accepted_parameters, first_supported

        config = self.config
        params = accepted_parameters(SFTConfig.__init__)
        has_eval = eval_data is not None
        args: dict[str, Any] = {
            "output_dir": config.output_dir,
            "num_train_epochs": config.num_epochs,
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "lr_scheduler_type": config.lr_scheduler_type,
            "optim": config.optim,
            "fp16": config.fp16,
            "bf16": config.bf16,
            "max_grad_norm": config.max_grad_norm,
            "gradient_checkpointing": config.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "logging_steps": config.logging_steps,
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,
            "push_to_hub": config.push_to_hub,
            "hub_model_id": config.hub_model_id,
            "hub_token": config.hub_token,
            "hub_private_repo": config.hub_private_repo,
            "seed": config.seed,
            "report_to": config.report_to,
        }
        # Renamed or removed across releases.
        if "warmup_ratio" in params:
            args["warmup_ratio"] = config.warmup_ratio
        else:  # transformers 5 takes a ratio in warmup_steps
            args["warmup_steps"] = config.warmup_ratio
        strategy_key = first_supported(params, "eval_strategy", "evaluation_strategy")
        if strategy_key:
            args[strategy_key] = "steps" if has_eval else "no"
        if has_eval:
            args["eval_steps"] = config.eval_steps
        length_key = first_supported(params, "max_length", "max_seq_length")
        if length_key:
            args[length_key] = config.max_seq_length
        if config.prompt_template != "messages":
            args["dataset_text_field"] = config.text_column

        unsupported = sorted(set(args) - params)
        if unsupported:
            if config.push_to_hub and "hub_private_repo" in unsupported:
                raise ValueError(
                    "This transformers version cannot create private Hub repositories "
                    "from the trainer. Upgrade it, or push with LegalTrainer.push_to_hub()."
                )
            logger.warning(f"Ignoring settings unsupported by this TRL version: {unsupported}")
            for key in unsupported:
                del args[key]

        trainer_params = accepted_parameters(SFTTrainer.__init__)
        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": SFTConfig(**args),
            "train_dataset": train_data,
            "eval_dataset": eval_data,
        }
        tokenizer_key = first_supported(trainer_params, "processing_class", "tokenizer")
        if tokenizer_key:
            trainer_kwargs[tokenizer_key] = self.tokenizer
        return SFTTrainer(**trainer_kwargs)

    def _torch_dtype(self, torch):
        if self.config.bf16:
            return torch.bfloat16
        if self.config.fp16:
            return torch.float16
        return torch.float32

    def _merge_into_16bit_base(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from legalkit._compat import dtype_kwargs

        dtype = torch.float16 if self.config.fp16 else torch.bfloat16
        with tempfile.TemporaryDirectory() as adapter_dir:
            self.model.save_pretrained(adapter_dir)
            base = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                trust_remote_code=self.config.trust_remote_code,
                **dtype_kwargs(dtype),
            )
            return PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

    def _count_parameters(self):
        """Count trainable and total parameters."""
        if hasattr(self.model, "get_nb_trainable_parameters"):
            return self.model.get_nb_trainable_parameters()
        trainable = 0
        total = 0
        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        return trainable, total


def _qlora_quantization(config: LegalTrainingConfig):
    from legalkit.finetune.adapters import create_qlora_config

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "QLoRA needs bitsandbytes and a CUDA GPU: pip install 'legal-llm-toolkit[qlora]'"
        ) from e
    return create_qlora_config(config)
