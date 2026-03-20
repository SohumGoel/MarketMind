"""
Unit tests for evaluation metrics — label extraction and coarse mapping.
"""

import pytest

from evaluation.metrics import extract_direction_from_output, coarse_label, compute_metrics


# --- coarse_label tests ---

def test_coarse_label_up():
    assert coarse_label("up_1_2") == "up"
    assert coarse_label("up_0_1") == "up"
    assert coarse_label("up_2_3") == "up"


def test_coarse_label_down():
    assert coarse_label("down_1_2") == "down"
    assert coarse_label("down_0_1") == "down"


def test_coarse_label_neutral():
    assert coarse_label("neutral") == "neutral"
    assert coarse_label("flat") == "neutral"


# --- extract_direction_from_output tests ---

def test_extract_direction_bullish_output():
    text = (
        "[Positive Developments]:\n1. Revenue grew 15%.\n\n"
        "[Potential Concerns]:\n1. Competition.\n\n"
        "[Prediction & Analysis]:\n"
        "The stock will likely rise next week driven by strong earnings."
    )
    result = extract_direction_from_output(text)
    assert result == "up"


def test_extract_direction_bearish_output():
    text = (
        "[Positive Developments]:\n1. None notable.\n\n"
        "[Potential Concerns]:\n1. Earnings miss, margin compression.\n\n"
        "[Prediction & Analysis]:\n"
        "The stock is expected to fall due to weak guidance."
    )
    result = extract_direction_from_output(text)
    assert result == "down"


def test_extract_direction_neutral_output():
    text = (
        "[Prediction & Analysis]:\n"
        "The market is expected to remain flat and stable this week."
    )
    result = extract_direction_from_output(text)
    assert result == "neutral"


def test_extract_direction_unknown():
    result = extract_direction_from_output("No directional signal here at all.")
    assert result == "unknown"


# --- compute_metrics tests ---

def test_compute_metrics_perfect():
    preds = ["up", "down", "neutral", "up"]
    refs = ["up", "down", "neutral", "up"]
    metrics = compute_metrics(preds, refs)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_compute_metrics_all_wrong():
    preds = ["down", "up", "up"]
    refs = ["up", "down", "down"]
    metrics = compute_metrics(preds, refs)
    assert metrics["accuracy"] == 0.0


def test_compute_metrics_returns_required_keys():
    metrics = compute_metrics(["up"], ["up"])
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "classification_report" in metrics
