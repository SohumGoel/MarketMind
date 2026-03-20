"""
Classification head ablation for MarketMind.

Adds a structured label token (e.g. "up_1_2") before the free-form analysis
in the model output. The combined loss is:

    L_total = L_SFT + lambda * L_cls

where L_cls is the cross-entropy loss on the label token prediction.

TODO (Sohum): Implement make_cls_formatting_func and ClsHeadCallback.
              The stubs below define the expected interface.
"""

from typing import Callable


def make_cls_formatting_func(base_formatting_func: Callable) -> Callable:
    """
    Wrap the base formatting function to prepend the label token.

    The label token (e.g. "Prediction: up_1_2") is inserted at the start
    of the assistant turn, before the [Positive Developments] section.

    Args:
        base_formatting_func: The base ChatML formatting function.

    Returns:
        A new formatting function that prepends the label token.
    """
    def formatting_func_with_cls(example: dict) -> str:
        base_text = base_formatting_func(example)
        label = example.get("label", "unknown")
        # Insert label token right after <|im_start|>assistant\n
        label_prefix = f"Prediction: {label}\n\n"
        return base_text.replace(
            "<|im_start|>assistant\n",
            f"<|im_start|>assistant\n{label_prefix}",
        )

    return formatting_func_with_cls


class ClsHeadCallback:
    """
    TODO (Sohum): Implement custom Trainer callback to apply the
    classification head loss weighting (cls_lambda).

    This will require subclassing transformers.TrainerCallback and
    overriding on_step_end to add the weighted cls loss to the
    standard SFT loss.
    """
    pass
