"""
Unit tests for data loading and prompt formatting.
"""

import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from data.formatting import format_example_qwen, _strip_llama_tags


# --- Fixtures ---

SAMPLE_EXAMPLE = {
    "prompt": "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nAnalyze AAPL stock. [/INST]",
    "answer": "[Positive Developments]:\n1. Strong iPhone sales.\n\n[Potential Concerns]:\n1. Supply chain risks.\n\n[Prediction & Analysis]:\nStock will likely go up.",
    "label": "up_1_2",
    "symbol": "AAPL",
    "period": "2024-01-01_2024-01-07",
}


# --- Formatting tests ---

def test_strip_llama_tags_removes_inst():
    raw = "[INST] Hello [/INST]"
    result = _strip_llama_tags(raw)
    assert "[INST]" not in result
    assert "[/INST]" not in result
    assert "Hello" in result


def test_strip_llama_tags_removes_sys():
    raw = "<<SYS>>\nSystem prompt here.\n<</SYS>>\nUser message."
    result = _strip_llama_tags(raw)
    assert "<<SYS>>" not in result
    assert "<</SYS>>" not in result
    assert "User message." in result


def test_format_example_qwen_structure():
    formatted = format_example_qwen(SAMPLE_EXAMPLE)

    assert "<|im_start|>system" in formatted
    assert "<|im_start|>user" in formatted
    assert "<|im_start|>assistant" in formatted
    assert "<|im_end|>" in formatted


def test_format_example_qwen_contains_answer():
    formatted = format_example_qwen(SAMPLE_EXAMPLE)
    assert "Strong iPhone sales" in formatted
    assert "Supply chain risks" in formatted


def test_format_example_qwen_no_llama_tags():
    formatted = format_example_qwen(SAMPLE_EXAMPLE)
    assert "[INST]" not in formatted
    assert "<<SYS>>" not in formatted


def test_format_example_qwen_order():
    formatted = format_example_qwen(SAMPLE_EXAMPLE)
    sys_pos = formatted.index("<|im_start|>system")
    user_pos = formatted.index("<|im_start|>user")
    asst_pos = formatted.index("<|im_start|>assistant")
    assert sys_pos < user_pos < asst_pos
