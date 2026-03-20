# MarketMind

MarketMind is a multi-agent LLM system that helps retail investors understand stock price movements using fine-tuned models, RAG, and live data retrieval. The **midpoint deliverable** is a fine-tuned Qwen3-8B model trained on the FinGPT Dow30 dataset with ablations across quantization formats and LoRA ranks.

---

## Repo Structure

```
MarketMind/
├── configs/
│   ├── base_config.yaml          # All defaults — edit here to change global settings
│   └── ablations/                # 8 ablation configs, each overrides base
├── data/
│   ├── dataset.py                # FinGPT Dow30 dataset loading + train/test split
│   └── formatting.py             # Prompt formatting (Llama [INST] -> Qwen3 ChatML)
├── training/
│   ├── train.py                  # Entry point: python training/train.py --config ...
│   └── sft_trainer.py            # Core fine-tuning logic (BnB, LoRA, SFTTrainer)
├── evaluation/
│   ├── evaluate.py               # Eval script: loads checkpoint, runs on test set
│   └── metrics.py                # Label extraction, coarse accuracy, F1
├── agents/                       # STUB ONLY — not needed for fine-tuning
│   ├── base_agent.py
│   ├── data_collection/          # news, SEC, price, sentiment agents (TODO: Sohum)
│   ├── retrieval/                # RAG pipeline, news clustering (TODO: Sohum/Yash)
│   └── synthesis/                # Evaluator agent wiring (TODO)
├── scripts/
│   ├── run_ablations.sh          # Run all 8 ablations sequentially
│   └── run_eval_all.sh           # Eval all checkpoints, write results_summary.json
├── notebooks/
│   ├── train_colab.ipynb         # ⭐ Self-contained Colab training notebook (start here)
│   └── explore_dataset.ipynb     # Self-contained Colab EDA notebook (run before training)
├── tests/
│   ├── test_dataset.py
│   └── test_metrics.py
├── outputs/                      # Gitignored — checkpoints written here at runtime
└── requirements.txt
```

---

## Colab Notebooks (Primary Workflow)

Training runs on Google Colab — `train_colab.ipynb` is the only notebook you need. It is fully self-contained: edit the config cell (Section 2) to pick an ablation, then run all cells top to bottom. Checkpoints save to Google Drive.

> `explore_dataset.ipynb` is an optional EDA notebook (label distribution, token length histogram) — not required for training.

---

## Setup (local dev only — not needed for training)

```bash
# 1. Clone the repo
git clone https://github.com/YOURHANDLE/MarketMind.git
cd MarketMind

# 2. Create conda environment (recommended on A100 nodes)
conda create -n marketmind python=3.11 -y
conda activate marketmind

# 3. Install dependencies
pip install -r requirements.txt

# 4. WandB login (for experiment tracking)
wandb login

# 5. HuggingFace login (needed to pull Qwen3 weights)
huggingface-cli login
```

**Verify GPU:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Running a Single Training Run

**Recommended starting point — QLoRA 4-bit, LoRA rank 32:**
```bash
python training/train.py --config configs/ablations/qlora_r32.yaml
```

**Smoke test (5 steps, ~2-3 min — verify pipeline before committing to a full run):**
```bash
python training/train.py --config configs/ablations/qlora_r32.yaml --smoke_test
```

**Full fine-tune ablation:**
```bash
python training/train.py --config configs/ablations/full_finetune.yaml
```

**With custom WandB project:**
```bash
python training/train.py --config configs/ablations/qlora_r32.yaml --wandb_project marketmind-ablations
```

Checkpoints are saved to `outputs/runs/{run_name}/` as defined in each ablation config.

---

## Running All Ablations

```bash
bash scripts/run_ablations.sh
```

Runs all 8 ablation configs sequentially. To parallelize across multiple GPUs, prefix each command with `CUDA_VISIBLE_DEVICES=N`.

---

## Evaluation

**Evaluate a fine-tuned checkpoint:**
```bash
python evaluation/evaluate.py \
    --checkpoint outputs/runs/qlora_4bit_r32/checkpoint-best \
    --split test
```

**Compare against zero-shot Qwen3-8B baseline:**
```bash
python evaluation/evaluate.py \
    --checkpoint outputs/runs/qlora_4bit_r32/checkpoint-best \
    --baseline
```

**Evaluate all checkpoints and aggregate results:**
```bash
bash scripts/run_eval_all.sh
# Writes: outputs/results_summary.json
```

---

## Ablation Matrix

| Run Name | Config File | Quant | LoRA Rank | ~VRAM |
|---|---|---|---|---|
| qlora_4bit_r16 | ablations/qlora_r16.yaml | 4-bit NF4 | 16 | ~18 GB |
| qlora_4bit_r32 | ablations/qlora_r32.yaml | 4-bit NF4 | 32 | ~20 GB |
| qlora_4bit_r64 | ablations/qlora_r64.yaml | 4-bit NF4 | 64 | ~24 GB |
| lora_8bit_r16 | ablations/lora_8bit_r16.yaml | 8-bit | 16 | ~28 GB |
| lora_8bit_r32 | ablations/lora_8bit_r32.yaml | 8-bit | 32 | ~32 GB |
| lora_8bit_r64 | ablations/lora_8bit_r64.yaml | 8-bit | 64 | ~38 GB |
| full_finetune | ablations/full_finetune.yaml | bf16 full | N/A | ~60-70 GB |
| cls_head_r32 | ablations/cls_head_r32.yaml | 4-bit NF4 | 32 | ~20 GB |

All runs log to WandB — compare training curves directly on the same project dashboard.

---

## Config System

All hyperparameters live in `configs/base_config.yaml`. Each ablation yaml overrides only the keys that differ:

```yaml
# configs/ablations/qlora_r32.yaml
run_name: "qlora_4bit_r32"
lora_r: 32
lora_alpha: 64
load_in_4bit: true
load_in_8bit: false
```

To change a global default (e.g., number of epochs), edit `base_config.yaml`. To add a new ablation, copy any existing ablation yaml and modify the relevant keys. The `run_name` field controls both the WandB run name and the output directory path (`outputs/runs/{run_name}/`).

---

## Verification Checklist

Run these in order before launching full ablations:

```bash
# 1. Dataset loads (catches HuggingFace auth issues early)
python -c "
from data.dataset import load_fingpt_dow30
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/base_config.yaml')
ds = load_fingpt_dow30(cfg)
print(ds)
"

# 2. Prompt formatting looks correct
python -c "
from data.dataset import load_fingpt_dow30
from data.formatting import format_example_qwen
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/base_config.yaml')
ds = load_fingpt_dow30(cfg)
print(format_example_qwen(ds['train'][0]))
"

# 3. Smoke test (5 steps end-to-end, ~2-3 min)
python training/train.py --config configs/ablations/qlora_r32.yaml --smoke_test

# 4. Evaluate baseline after first full run
python evaluation/evaluate.py \
    --checkpoint outputs/runs/qlora_4bit_r32/checkpoint-best \
    --baseline
```

---

## Work Division

| Area | Owner | Status at Midpoint |
|---|---|---|
| Fine-tuning training loop | Yash | Core midpoint deliverable |
| Evaluation pipeline | Yash | Core midpoint deliverable |
| News embedding/clustering | Yash | Post-midpoint |
| Model hosting | Yash | Post-midpoint |
| Data collection agents | Sohum | Stub (post-midpoint) |
| RAG retrieval pipeline | Sohum | Stub (post-midpoint) |
| Classification head ablation | Sohum | Post-midpoint |
| UI (Streamlit) | TBD | End-of-semester |

> **Note for Yash:** The `agents/` directory is scaffold only and has no runnable code. It does not affect training in any way — ignore it entirely for the midpoint.
