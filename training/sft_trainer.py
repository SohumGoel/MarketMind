"""
Core fine-tuning logic for MarketMind.

Handles:
- BitsAndBytes quantization config (4-bit NF4, 8-bit, or none for full fine-tune)
- LoRA / QLoRA setup via PEFT
- ClsHeadSFTTrainer instantiation (SFT + classification loss)
"""

import logging
import torch
import shutil
from pathlib import Path

from peft import get_peft_model
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig

from training.classification_head import ClsHeadSFTTrainer
from data.dataset import load_fingpt_dow30, make_cls_formatting_func
from data.formatting import format_example_qwen

from training.debug_cls import (
    debug_cls_tokens,
    debug_cls_dataset,
    debug_one_training_batch,
    debug_eval_schedule,
    print_trainable_params,
)

log = logging.getLogger(__name__)


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


def _build_peft_config(cfg: DictConfig, cls_token_ids: list[int] | None = None):
    """Build LoRA config. Returns None if use_peft is false."""
    if not cfg.use_peft:
        return None

    from peft import LoraConfig, TaskType

    peft_kwargs = dict(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Train only these embedding rows, not the full embedding matrix
    if False and cls_token_ids:
        peft_kwargs["trainable_token_indices"] = {
            "embed_tokens": cls_token_ids
        }

    return LoraConfig(**peft_kwargs)


def build_and_train(cfg: DictConfig) -> None:
    """
    Build model, tokenizer, and trainer from config, then run training.

    Args:
        cfg: Merged OmegaConf config (base + ablation override).
    """
    print("Starting run: %s", cfg.run_name)

    # --- Quantization config ---
    bnb_config = _build_bnb_config(cfg)

    # --- Load model ---
    model_kwargs = dict(
        pretrained_model_name_or_path=cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    print("Loading model: %s", cfg.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    format_fn = make_cls_formatting_func(format_example_qwen)

    dataset, meta = load_fingpt_dow30(
        cfg,
        formatting_func=format_fn,
    )

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": meta["all_label_tokens"]}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    print("[train] Class tokens and ids:")
    for tok in meta["all_label_tokens"]:
        ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
        print(f"  {tok} -> {ids}")

    cls_token_ids = [
        tokenizer.convert_tokens_to_ids(tok)
        for tok in meta["all_label_tokens"]
    ]

    print("[debug trainable params for embeddings/head]")
    for name, p in model.named_parameters():
        if "embed" in name.lower() or "lm_head" in name.lower():
            print(name, p.requires_grad, tuple(p.shape))
            assert p.requires_grad, f"Expected embedding and head params to be trainable, but {name} is frozen."

    # Verify cls tokens are valid (not mapped to unk)
    unk_id = tokenizer.unk_token_id
    token_table = {
        tok: tid for tok, tid in zip(meta["all_label_tokens"], cls_token_ids)
    }
    print("cls token set (%d tokens):", len(token_table))
    for tok, tid in sorted(token_table.items(), key=lambda x: x[1]):
        flag = " *** UNK ***" if tid == unk_id else ""
        print("  %s -> %d%s", tok, tid, flag)

    cls_token_id_set = set(cls_token_ids)

    num_missing = 0
    for ex in dataset["train"]:
        ids = tokenizer(
            ex["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
        )["input_ids"]
        if not any(tok in cls_token_id_set for tok in ids):
            num_missing += 1

    print("cls tokens missing after truncation: %d / %d", num_missing, len(dataset["train"]))
    
    peft_config = _build_peft_config(cfg, cls_token_ids=cls_token_ids)

    if bnb_config is not None and cfg.use_peft:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    
    if peft_config is not None:
        model = get_peft_model(model, peft_config)

    # compute_dtype = torch.bfloat16
    # model.get_input_embeddings().to(compute_dtype)
    # model.lm_head.to(compute_dtype)

    model.get_input_embeddings().weight.requires_grad = True
    model.lm_head.weight.requires_grad = True

    training_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_strategy=cfg.logging_strategy,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        completion_only_loss=False,
    )

    if hasattr(cfg, "max_steps"):
        training_kwargs["max_steps"] = cfg.max_steps

    sft_config = SFTConfig(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    # if peft_config is not None:
    #     trainer_kwargs["peft_config"] = peft_config

    trainer = ClsHeadSFTTrainer(
        **trainer_kwargs,
        cls_lambda=cfg.cls_lambda,
        cls_token_ids=cls_token_ids,
    )

    debug_cls_tokens(tokenizer, meta["all_label_tokens"])
    debug_cls_dataset(
        dataset["train"],
        tokenizer,
        cls_token_ids=cls_token_ids,
        max_length=cfg.max_seq_length,
        n_show=5,
    )

    debug_eval_schedule(cfg)
    debug_one_training_batch(trainer, tokenizer, n_batches=1, topk=10)

    print_trainable_params(trainer.model)

    print("Starting training...")
    trainer.train()

    # --- Save final export (model/tokenizer only) ---
    final_dir = Path(cfg.final_dir.format(run_name=cfg.run_name))
    final_dir.mkdir(parents=True, exist_ok=True)

    print("Saving final model to %s", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # --- Cleanup training checkpoints ---
    run_dir = Path(cfg.output_dir.format(run_name=cfg.run_name))

    if run_dir.exists():
        print("Deleting training artifacts at %s", run_dir)
        shutil.rmtree(run_dir)
    else:
        print("Expected training artifacts at %s not found for cleanup.", run_dir)

    print("Done. Run: %s", cfg.run_name)
