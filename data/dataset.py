"""
FinGPT Dow30 dataset loading and preprocessing.

Dataset: FinGPT/fingpt-forecaster-dow30-202305-202312
Columns: prompt, answer, period, symbol, label
"""

from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig


DATASET_ID = "FinGPT/fingpt-forecaster-dow30-202305-202312"


def load_fingpt_dow30(cfg: DictConfig) -> DatasetDict:
    """
    Load the FinGPT Dow30 forecaster dataset and split into train/test.

    Args:
        cfg: OmegaConf config with fields:
            - dataset_name (str): HuggingFace dataset ID
            - test_split_size (float): fraction for test split
            - seed (int): random seed for reproducibility

    Returns:
        DatasetDict with 'train' and 'test' keys.
    """
    dataset_id = getattr(cfg, "dataset_name", DATASET_ID)
    ds = load_dataset(dataset_id, split="train")
    split = ds.train_test_split(
        test_size=cfg.test_split_size,
        seed=cfg.seed,
    )
    return split
