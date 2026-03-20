"""
Core fine-tuning logic for MarketMind.

Handles:
- BitsAndBytes quantization config (4-bit NF4, 8-bit, or none for full fine-tune)
- LoRA / QLoRA setup via PEFT
- SFTTrainer instantiation from trl
- Classification head ablation (use_cls_head=true in config)
"""

import os
import torch

from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from data.dataset import load_fingpt_dow30
from data.formatting import format_example_qwen


def _build_bnb_config(cfg: DictConfig):
    """Build BitsAndBytes quantization config. Returns None for full fine-tune."""
    from transformers import BitsAndBytesConfig

    if cfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )
    elif cfg.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None  # full fine-tune: no quantization


def _build_peft_config(cfg: DictConfig):
    """Build LoRA config. Returns None if use_peft is false."""
    if not cfg.use_peft:
        return None

    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_and_train(cfg: DictConfig) -> None:
    """
    Build model, tokenizer, and trainer from config, then run training.

    Args:
        cfg: Merged OmegaConf config (base + ablation override).
    """
    print(f"[train] Starting run: {cfg.run_name}")

    # --- Quantization config ---
    bnb_config = _build_bnb_config(cfg)
    peft_config = _build_peft_config(cfg)

    # --- Load model ---
    model_kwargs = dict(
        pretrained_model_name_or_path=cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    print(f"[train] Loading model: {cfg.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Prepare model for k-bit training (required for QLoRA/LoRA on quantized models)
    if bnb_config is not None and cfg.use_peft:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Load dataset ---
    print("[train] Loading dataset...")
    dataset = load_fingpt_dow30(cfg)

    # --- SFTConfig (TrainingArguments for trl) ---
    training_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        max_seq_length=cfg.max_seq_length,
    )

    # Smoke test: override max_steps if set
    if hasattr(cfg, "max_steps"):
        training_kwargs["max_steps"] = cfg.max_steps

    sft_config = SFTConfig(**training_kwargs)

    # --- Trainer ---
    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        formatting_func=format_example_qwen,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    # Classification head ablation: wrap formatting_func with label token prefix
    if cfg.use_cls_head:
        from training.classification_head import make_cls_formatting_func, ClsHeadCallback
        trainer_kwargs["formatting_func"] = make_cls_formatting_func(format_example_qwen)
        # cls_lambda applied via custom callback/loss — see classification_head.py
        print(f"[train] Classification head enabled (lambda={cfg.cls_lambda})")

    trainer = SFTTrainer(**trainer_kwargs)

    print("[train] Starting training...")
    trainer.train()

    # --- Save ---
    output_dir = cfg.output_dir
    print(f"[train] Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train] Done. Run: {cfg.run_name}")
