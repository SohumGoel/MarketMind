"""
Evaluation metrics for MarketMind fine-tuned model.

Two evaluation modes:
1. Coarse 3-class (up / down / neutral): extracted from free-form model output.
2. Fine-grained FinGPT label (up_1_2, down_0_1, etc.): for cls_head ablation.
"""

import re
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Maps directional keywords from generated text -> coarse label
_DIRECTION_KEYWORDS = {
    "up": "up",
    "increase": "up",
    "rise": "up",
    "rising": "up",
    "higher": "up",
    "bullish": "up",
    "positive": "up",
    "growth": "up",
    "down": "down",
    "decrease": "down",
    "fall": "down",
    "falling": "down",
    "lower": "down",
    "bearish": "down",
    "negative": "down",
    "decline": "down",
    "hold": "neutral",
    "flat": "neutral",
    "neutral": "neutral",
    "unchanged": "neutral",
    "stable": "neutral",
}


def extract_direction_from_output(generated_text: str) -> str:
    """
    Extract predicted stock direction from model's free-form output.

    Looks for the [Prediction & Analysis] section first, then falls back
    to scanning the last 300 characters for directional keywords.

    Returns one of: 'up', 'down', 'neutral', 'unknown'
    """
    # Primary: look for explicit Prediction field
    match = re.search(
        r"\[Prediction\s*&\s*Analysis\][:\s]*.*?(?:will|likely|expected to|predict|forecast)?\s*(\w+)",
        generated_text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        word = match.group(1).lower()
        if word in _DIRECTION_KEYWORDS:
            return _DIRECTION_KEYWORDS[word]

    # Fallback: scan the last 300 characters for keywords
    tail = generated_text[-300:].lower()
    for keyword, direction in _DIRECTION_KEYWORDS.items():
        if re.search(r"\b" + keyword + r"\b", tail):
            return direction

    return "unknown"


def coarse_label(fingpt_label: str) -> str:
    """
    Map a FinGPT fine-grained label to coarse direction.

    Examples:
        'up_1_2'    -> 'up'
        'down_0_1'  -> 'down'
        'up_0_1'    -> 'up'
        'neutral'   -> 'neutral'

    Args:
        fingpt_label: Raw label string from dataset 'label' field.

    Returns:
        One of: 'up', 'down', 'neutral'
    """
    label = fingpt_label.strip().lower()
    if label.startswith("up"):
        return "up"
    elif label.startswith("down"):
        return "down"
    return "neutral"


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """
    Compute accuracy and macro F1 for directional classification.

    Args:
        predictions: List of coarse predicted labels ('up'/'down'/'neutral'/'unknown').
        references: List of coarse ground-truth labels.

    Returns:
        dict with 'accuracy', 'f1_macro', 'classification_report'
    """
    labels = ["up", "down", "neutral"]

    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, labels=labels, average="macro", zero_division=0)
    report = classification_report(references, predictions, labels=labels, zero_division=0)

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1, 4),
        "classification_report": report,
    }
