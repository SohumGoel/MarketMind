"""
MarketMind fine-tuning entry point.

Usage:
    python training/train.py --config configs/ablations/qlora_r32.yaml
    python training/train.py --config configs/ablations/qlora_r32.yaml --smoke_test
    python training/train.py --config configs/ablations/full_finetune.yaml --wandb_project marketmind
"""

import argparse
import os

from omegaconf import OmegaConf

from training.sft_trainer import build_and_train


def parse_args():
    parser = argparse.ArgumentParser(description="MarketMind fine-tuning")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to ablation config yaml (merged on top of base_config.yaml)",
    )
    parser.add_argument(
        "--wandb_project",
        default="marketmind",
        help="WandB project name",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run only 5 steps to verify the pipeline end-to-end (~2-3 min)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_cfg = OmegaConf.load("configs/base_config.yaml")
    override_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, override_cfg)

    # Smoke test: override to 5 steps
    if args.smoke_test:
        cfg.num_train_epochs = 1
        cfg.max_steps = 5
        cfg.logging_steps = 1
        cfg.eval_steps = 5
        cfg.save_steps = 5
        cfg.run_name = cfg.run_name + "_smoke"
        print("[smoke_test] Overriding to 5 steps.")

    os.environ["WANDB_PROJECT"] = args.wandb_project

    build_and_train(cfg)


if __name__ == "__main__":
    main()
