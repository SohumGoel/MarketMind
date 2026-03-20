#!/usr/bin/env bash
# Run all 8 MarketMind ablation training runs sequentially.
# To parallelize across multiple GPUs:
#   CUDA_VISIBLE_DEVICES=0 python training/train.py --config ... &
#   CUDA_VISIBLE_DEVICES=1 python training/train.py --config ... &
set -e

CONFIGS=(
    "configs/ablations/qlora_r16.yaml"
    "configs/ablations/qlora_r32.yaml"
    "configs/ablations/qlora_r64.yaml"
    "configs/ablations/lora_8bit_r16.yaml"
    "configs/ablations/lora_8bit_r32.yaml"
    "configs/ablations/lora_8bit_r64.yaml"
    "configs/ablations/full_finetune.yaml"
    "configs/ablations/cls_head_r32.yaml"
)

PROJECT=${WANDB_PROJECT:-"marketmind-ablations"}

echo "Starting MarketMind ablation sweep (${#CONFIGS[@]} runs)"
echo "WandB project: $PROJECT"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running: $CONFIG"
    echo "=========================================="
    python training/train.py --config "$CONFIG" --wandb_project "$PROJECT"
    echo ""
done

echo "All ablations complete."
