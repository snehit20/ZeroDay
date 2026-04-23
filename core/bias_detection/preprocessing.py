"""
AEGIS — Data Preprocessing
=============================

Handles missing value imputation and categorical encoding.
Produces clean, numeric data ready for bias detection and model training.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..config import get_logger

logger = get_logger(__name__)


def preprocess_dataset(
    df: pd.DataFrame,
    target: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess a dataset for the AEGIS pipeline.

    Steps:
        1. Handle missing values (mean for numeric, mode for categorical)
        2. Label-encode categorical columns

    Args:
        df: Raw input DataFrame.
        target: Optional target column name (excluded from auto-detection).
        categorical_cols: Explicit list of categorical columns.
            If None, auto-detected from dtype.
        numerical_cols: Explicit list of numerical columns.
            If None, auto-detected from dtype.

    Returns:
        Tuple of (processed DataFrame, metadata dict).
    """
    logger.info("Preprocessing dataset...")
    df = df.copy()

    # Auto-detect column types if not provided
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # --- 1. Handle Missing Values ---
    imputation_info = {}

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            imputation_info[col] = {"method": "mean", "value": float(mean_val)}

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            imputation_info[col] = {"method": "mode", "value": str(mode_val)}

    # --- 2. Encode Categorical Columns ---
    encoders = {}
    encoded_cols = []

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            encoded_cols.append(col)

    metadata = {
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "encoded_columns": encoded_cols,
        "encoders": encoders,
        "imputation": imputation_info,
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

    logger.info(
        f"Preprocessing complete: {len(df)} rows, "
        f"{len(encoded_cols)} columns encoded, "
        f"{len(imputation_info)} columns imputed"
    )

    return df, metadata
