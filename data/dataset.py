from __future__ import annotations

import re
from typing import Callable, Dict, Tuple

from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig


DATASET_ID = "FinGPT/fingpt-forecaster-dow30-202305-202405"


def _normalize_label(label: str) -> str:
    s = label.strip().lower()
    s = s.replace("%", "")
    s = s.replace("-", "_")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _make_label_token(label: str) -> str:
    return f"<cls_{_normalize_label(label)}>"


def _build_label_token_map(ds: DatasetDict) -> Dict[str, str]:
    labels = sorted(set(ds["train"]["label"]) | set(ds["test"]["label"]))
    return {label: _make_label_token(label) for label in labels}


# -----------------------------
# Formatting hook
# -----------------------------

def make_cls_formatting_func(base_formatting_func: Callable) -> Callable:
    def formatting_func_with_cls(example: dict) -> str:
        base_text = base_formatting_func(example)
        label_tok = example["label_token"]
        label_prefix = f"Prediction: {label_tok}\n\n"

        return base_text.replace(
            "<|im_start|>assistant\n",
            f"<|im_start|>assistant\n{label_prefix}",
            1,
        )

    return formatting_func_with_cls

def load_fingpt_dow30(cfg, formatting_func=None):
    ds = load_dataset(DATASET_ID)
    ds = DatasetDict({"train": ds["train"], "test": ds["test"]})

    label_to_token = _build_label_token_map(ds)

    def add_fields(example):
        out = dict(example)
        out["completion"] = out.get("answer", "")
        out["label_token"] = label_to_token[out["label"]]
        return out

    ds["train"] = ds["train"].map(add_fields)
    ds["test"] = ds["test"].map(add_fields)

    if formatting_func is not None:
        ds["train"] = ds["train"].map(lambda ex: {"text": formatting_func(ex)})
        ds["test"] = ds["test"].map(lambda ex: {"text": formatting_func(ex)})

        # critical: remove prompt/completion/etc so TRL sees plain text dataset
        ds["train"] = ds["train"].remove_columns(
            [c for c in ds["train"].column_names if c != "text"]
        )
        ds["test"] = ds["test"].remove_columns(
            [c for c in ds["test"].column_names if c != "text"]
        )

    meta = {
        "label_to_token": label_to_token,
        "all_label_tokens": list(label_to_token.values()),
    }

    return ds, meta