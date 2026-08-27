import logging
import os

import pandas as pd

from src.config_loader import load_config
from src.data_contracts import DataContractError, validate_processed, validate_raw

logger = logging.getLogger(__name__)

TOY_DATA_ENV_VAR = "HOTEL_MLOPS_USE_TOY_DATA"


def use_toy_data() -> bool:
    """Whether the fast, DVC-tracked ~1000-row toy dataset should be used.

    Set HOTEL_MLOPS_USE_TOY_DATA=true to run the full pipeline end-to-end in
    seconds instead of against the full dataset, e.g. `make train-toy`.
    """
    return os.getenv(TOY_DATA_ENV_VAR, "false").strip().lower() in ("1", "true", "yes")


def load_and_clean_data(filepath: str, min_class_count: int) -> pd.DataFrame:
    """
    Loads dataset, validates it against the raw data contract, normalizes
    column names, handles basic data types, prepares dates, and validates
    the result against the processed data contract.

    Args:
        filepath (str): Path to the raw dataset.
        min_class_count (int): Minimum occurrences for a target class to be kept.

    Returns:
        pd.DataFrame: Cleaned dataframe, guaranteed to satisfy ProcessedBookingSchema.
    """
    try:
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Fail fast: reject malformed input before spending time cleaning it.
        df = validate_raw(df)

        # Drop duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        if initial_shape[0] != df.shape[0]:
            logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

        # Drop rows with missing target variable
        df = df.dropna(subset=["market_segment"])

        # Drop rows with a negative average daily rate (data entry errors, not
        # legitimate bookings) - the processed contract requires adr >= 0.
        invalid_adr = df["adr"] < 0
        if invalid_adr.any():
            logger.info(f"Removed {invalid_adr.sum()} row(s) with negative adr.")
            df = df[~invalid_adr]

        # Filter rare target classes to avoid errors during stratified splitting
        target_counts = df["market_segment"].value_counts()
        valid_classes = target_counts[target_counts >= min_class_count].index
        df = df[df["market_segment"].isin(valid_classes)]

        # Convert string months to integers
        month_map = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }

        if "arrival_date_month" in df.columns:
            df["month"] = df["arrival_date_month"].map(month_map)
            df = df.drop(columns=["arrival_date_month"])

        # Fail fast: reject the cleaned output before it reaches Optuna/SMOTE/training.
        df = validate_processed(df)

        logger.info(f"Data loaded and cleaned successfully. Final shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        logger.error(f"Dataset not found at {filepath}: {e}")
        raise
    except DataContractError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data cleaning: {e}")
        raise


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Clean the raw dataset into the processed CSV.")
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Use the ~1000-row DVC-tracked toy dataset instead of the full dataset "
        f"(equivalent to {TOY_DATA_ENV_VAR}=true).",
    )
    args = parser.parse_args()

    config = load_config()

    if args.toy or use_toy_data():
        raw_path = config["data"]["toy_raw_data_path"]
        processed_path = config["data"]["toy_processed_data_path"]
        logger.info(f"Using the toy dataset at {raw_path}")
    else:
        raw_path = config["data"]["raw_data_path"]
        processed_path = config["data"]["processed_data_path"]

    min_count = config["data"]["valid_classes_min_count"]

    clean_df = load_and_clean_data(raw_path, min_count)
    clean_df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")
