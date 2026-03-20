"""
Prompt formatting for the FinGPT Dow30 dataset.

The raw dataset uses Llama-2 [INST] / <<SYS>> tags.
This module converts each example to Qwen3's ChatML format
(<|im_start|>system/user/assistant<|im_end|>) for SFTTrainer.

Usage with trl SFTTrainer:
    from data.formatting import format_example_qwen
    trainer = SFTTrainer(..., formatting_func=format_example_qwen)
"""

import re


SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Your task is to list the positive "
    "developments and potential concerns for companies based on relevant news and "
    "basic financial data from the past weeks, then make a prediction about the "
    "companies' stock price movement for the upcoming week. "
    "Your answer format should be as follows:\n\n"
    "[Positive Developments]:\n1. ...\n\n"
    "[Potential Concerns]:\n1. ...\n\n"
    "[Prediction & Analysis]:\n..."
)


def _strip_llama_tags(text: str) -> str:
    """Remove Llama-2 [INST], [/INST], <<SYS>>, <</SYS>> tags from raw prompt."""
    text = re.sub(r"\[/?INST\]", "", text)
    text = re.sub(r"<<SYS>>.*?<</SYS>>", "", text, flags=re.DOTALL)
    return text.strip()


def format_example_qwen(example: dict) -> str:
    """
    Format a single FinGPT dataset example as a Qwen3 ChatML string.

    SFTTrainer with formatting_func receives the full string (prompt + completion)
    and handles prompt masking internally.

    Args:
        example: dict with keys 'prompt' and 'answer'

    Returns:
        Full ChatML-formatted string for SFT training.
    """
    user_content = _strip_llama_tags(example["prompt"])
    assistant_content = example["answer"].strip()

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )
