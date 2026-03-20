"""
Evaluation script for MarketMind fine-tuned checkpoints.

Usage:
    # Evaluate a fine-tuned checkpoint
    python evaluation/evaluate.py \
        --checkpoint outputs/runs/qlora_4bit_r32/checkpoint-best \
        --split test

    # Compare against zero-shot Qwen3-8B baseline (no adapter)
    python evaluation/evaluate.py \
        --checkpoint outputs/runs/qlora_4bit_r32/checkpoint-best \
        --baseline
"""

import argparse
import json
import os

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.dataset import load_fingpt_dow30
from data.formatting import format_example_qwen, SYSTEM_PROMPT
from evaluation.metrics import extract_direction_from_output, coarse_label, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a MarketMind checkpoint")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (adapter weights)",
    )
    parser.add_argument(
        "--config",
        default="configs/base_config.yaml",
        help="Base config yaml (used for dataset loading params)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate zero-shot base model without adapter (for comparison)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (for quick sanity checks)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_dir: str, baseline: bool):
    """
    Load model and tokenizer.

    If baseline=True, loads the raw base model (no adapter).
    If baseline=False, loads the base model + LoRA adapter from checkpoint.
    """
    # Read model name from checkpoint's config if available
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path) and not baseline:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-8B")
    else:
        base_model_name = "Qwen/Qwen3-8B"

    print(f"[eval] Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if not baseline:
        print(f"[eval] Loading adapter from: {checkpoint_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        model = model.merge_and_unload()

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    return model, tokenizer


def generate_prediction(model, tokenizer, example: dict, max_new_tokens: int) -> str:
    """Run greedy decoding on a single example and return the generated text."""
    # Build prompt-only (no assistant turn) for generation
    user_content = example["prompt"]
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = load_fingpt_dow30(cfg)
    eval_data = dataset[args.split]

    if args.max_samples is not None:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.baseline)

    predictions = []
    references = []

    print(f"[eval] Evaluating {len(eval_data)} examples...")
    for example in tqdm(eval_data):
        generated = generate_prediction(model, tokenizer, example, args.max_new_tokens)
        pred_direction = extract_direction_from_output(generated)
        ref_direction = coarse_label(example["label"])

        predictions.append(pred_direction)
        references.append(ref_direction)

    results = compute_metrics(predictions, references)
    mode = "baseline" if args.baseline else "finetuned"
    results["mode"] = mode
    results["checkpoint"] = args.checkpoint
    results["n_samples"] = len(eval_data)

    print("\n" + "=" * 50)
    print(f"Results ({mode})")
    print("=" * 50)
    print(f"Accuracy:  {results['accuracy']}")
    print(f"F1 Macro:  {results['f1_macro']}")
    print(results["classification_report"])

    # Save results alongside checkpoint
    output_path = os.path.join(args.checkpoint, f"results_{mode}.json")
    with open(output_path, "w") as f:
        # Don't serialize the full classification_report string to JSON
        json_results = {k: v for k, v in results.items() if k != "classification_report"}
        json.dump(json_results, f, indent=2)
    print(f"[eval] Results saved to {output_path}")


if __name__ == "__main__":
    main()
