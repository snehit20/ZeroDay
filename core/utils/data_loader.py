"""
AEGIS — Dataset Loader
========================

Handles loading datasets from multiple sources:
    1. Pandas DataFrame (pass-through)
    2. CSV file path
    3. UCI Adult dataset (auto-download or local)
    4. Synthetic biased dataset (for testing)
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..config import get_logger, RAW_DATA_DIR

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    source: Union[pd.DataFrame, str, None] = None,
) -> pd.DataFrame:
    """
    Load a dataset from the specified source.

    Args:
        source: One of:
            - ``pd.DataFrame``: returned as-is
            - ``str``: treated as a CSV file path
            - ``None``: loads the UCI Adult Income dataset (auto-download)

    Returns:
        A pandas DataFrame ready for pipeline processing.

    Raises:
        FileNotFoundError: If a file path is given but doesn't exist.
        ValueError: If the source type is unsupported.
    """
    if isinstance(source, pd.DataFrame):
        logger.info(f"Dataset provided as DataFrame: {source.shape}")
        return source.copy()

    if isinstance(source, str):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {source}")
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset from {source}: {df.shape}")
        return df

    if source is None:
        return load_adult_dataset()

    raise ValueError(
        f"Unsupported dataset source type: {type(source).__name__}. "
        "Expected DataFrame, file path string, or None."
    )


def load_adult_dataset() -> pd.DataFrame:
    """
    Load the UCI Adult Income dataset.

    Tries in order:
        1. Local file at ``data/raw/adult.csv``
        2. ``sklearn.datasets.fetch_openml``
        3. Synthetic fallback

    Returns:
        Adult income DataFrame with standardised column names.
    """
    # --- Try local file ---
    local_path = RAW_DATA_DIR / "adult.csv"
    if local_path.exists():
        logger.info(f"Loading Adult dataset from {local_path}")
        df = pd.read_csv(local_path)
        return _standardise_adult(df)

    # Also check for labeled_data.csv which may be the adult dataset
    labeled_path = RAW_DATA_DIR / "labeled_data.csv"
    if labeled_path.exists():
        df = pd.read_csv(labeled_path, nrows=5)
        # Only use if it looks like the adult dataset
        adult_markers = {"age", "income", "race", "sex", "gender"}
        if adult_markers.intersection(set(c.lower() for c in df.columns)):
            logger.info(f"Loading dataset from {labeled_path}")
            return pd.read_csv(labeled_path)

    # --- Try sklearn download ---
    try:
        from sklearn.datasets import fetch_openml
        logger.info("Downloading Adult dataset via sklearn...")
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = data.frame
        # Save locally for future use
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
        logger.info(f"Adult dataset saved to {local_path}")
        return _standardise_adult(df)
    except Exception as e:
        logger.warning(f"Could not download Adult dataset: {e}")

    # --- Fallback to synthetic ---
    logger.warning("Using synthetic dataset as fallback")
    return create_synthetic_dataset()


def _standardise_adult(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise Adult dataset column names."""
    rename_map = {
        "education-num": "education_num",
        "marital-status": "marital_status",
        "capital-gain": "capital_gain",
        "capital-loss": "capital_loss",
        "hours-per-week": "hours_per_week",
        "native-country": "native_country",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Clean income column if present
    if "income" in df.columns:
        df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    return df


def create_synthetic_dataset(
    n: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic dataset with deliberate bias for demo/testing.

    The dataset simulates a loan-approval scenario with:
        - Gender bias (males approved more often)
        - Race bias (group_A approved more often)
        - Proxy feature (zip_code correlated with race)

    Args:
        n: Number of samples.
        seed: Random seed for reproducibility.

    Returns:
        Biased synthetic DataFrame.
    """
    rng = np.random.RandomState(seed)

    gender = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    race = rng.choice(
        ["group_A", "group_B", "group_C"], size=n, p=[0.6, 0.25, 0.15]
    )

    age = rng.normal(35, 10, n).clip(18, 65).astype(int)
    income = rng.normal(50000, 15000, n).clip(15000, 150000)
    credit_score = rng.normal(650, 80, n).clip(300, 850).astype(int)

    # Proxy: zip_code correlated with race
    zip_base = {"group_A": 10000, "group_B": 20000, "group_C": 30000}
    zip_code = np.array([zip_base[r] + rng.randint(0, 100) for r in race])

    # Biased target
    approval_prob = 0.3 * np.ones(n)
    approval_prob[gender == "male"] += 0.2
    approval_prob[race == "group_A"] += 0.15
    approval_prob += (credit_score - 500) / 2000
    approval_prob += (income - 30000) / 200000
    approval_prob = np.clip(approval_prob, 0.05, 0.95)

    approved = (rng.random(n) < approval_prob).astype(int)

    logger.info(f"Created synthetic dataset with {n} samples")

    return pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "zip_code": zip_code,
        "gender": gender,
        "race": race,
        "approved": approved,
    })
