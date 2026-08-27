"""Builds the small, fixed, deterministic sample of the raw dataset used for
fast end-to-end local runs (`make data-toy` / `make train-toy`).

The sample is stratified per market_segment class (capped at an even quota
per class) rather than proportional to the real class distribution, so that
every class surviving `valid_classes_min_count` still has enough rows for a
stratified train/test split and for SMOTE to oversample from. The output is
tracked with DVC, not committed as a plain file, so it stays reproducible and
content-addressed just like the full dataset.
"""

import logging
import math
import os

import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)


def make_toy_dataset(
    raw_path: str,
    output_path: str,
    target_column: str,
    sample_size: int,
    min_class_count: int,
    random_state: int,
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    counts = df[target_column].value_counts()
    valid_classes = counts[counts >= min_class_count].index
    df = df[df[target_column].isin(valid_classes)]

    per_class_quota = math.ceil(sample_size / len(valid_classes))
    sampled = df.groupby(target_column, group_keys=False)[df.columns].apply(
        lambda g: g.sample(n=min(len(g), per_class_quota), random_state=random_state)
    )
    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sampled.to_csv(output_path, index=False)
    logger.info(
        f"Toy dataset written to {output_path}: {len(sampled)} rows across "
        f"{sampled[target_column].nunique()} classes."
    )
    return sampled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    make_toy_dataset(
        raw_path=config["data"]["raw_data_path"],
        output_path=config["data"]["toy_raw_data_path"],
        target_column=config["model"]["target_column"],
        sample_size=config["data"]["toy_sample_size"],
        min_class_count=config["data"]["valid_classes_min_count"],
        random_state=config["model"]["random_state"],
    )
