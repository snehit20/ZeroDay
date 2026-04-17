"""
AEGIS -- Utility Functions
===========================

Shared helper functions used across the Bias Mitigation Engine.
Provides logging, validation, safe imports, and data transformation utilities.
"""

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and returns a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Safe Import Helpers
# ---------------------------------------------------------------------------

def safe_import(module_name: str) -> Optional[Any]:
    """
    Attempt to import a module, returning None if unavailable.

    Args:
        module_name: Fully qualified module name to import.

    Returns:
        The imported module or None if import fails.
    """
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        return None


def check_dependency(module_name: str, feature_name: str) -> bool:
    """
    Check if an optional dependency is available and log a warning if not.

    Args:
        module_name: Module to check.
        feature_name: Human-readable feature name for the warning message.

    Returns:
        True if available, False otherwise.
    """
    logger = get_logger("utils")
    mod = safe_import(module_name)
    if mod is None:
        logger.warning(
            f"Optional dependency '{module_name}' not found. "
            f"'{feature_name}' will use fallback implementation."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    context: str = "input",
) -> None:
    """
    Validate that a DataFrame meets basic requirements.

    Args:
        df: The DataFrame to validate.
        required_columns: List of column names that must be present.
        context: Description of where this data came from (for error messages).

    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame for {context}, got {type(df).__name__}")
    if df.empty:
        raise ValueError(f"DataFrame for {context} is empty.")
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame for {context} is missing required columns: {sorted(missing)}"
            )


def validate_bias_report(bias_report: Dict[str, Any]) -> None:
    """
    Validate the structure of a bias report from Phase 1.

    Args:
        bias_report: The bias report dictionary.

    Raises:
        ValueError: If required sections are missing.
    """
    if not isinstance(bias_report, dict):
        raise ValueError(f"bias_report must be a dict, got {type(bias_report).__name__}")

    expected_sections = [
        "distribution_bias",
        "outcome_bias",
        "fairness_metrics",
    ]
    missing = [s for s in expected_sections if s not in bias_report]
    if missing:
        logger = get_logger("utils")
        logger.warning(
            f"Bias report is missing sections: {missing}. "
            "Some bias classifications may be incomplete."
        )


# ---------------------------------------------------------------------------
# Data Transformation Utilities
# ---------------------------------------------------------------------------

def encode_categorical_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    drop_first: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    One-hot encode categorical features, returning the transformed DataFrame
    and a mapping of original column -> generated columns.

    Args:
        df: Input DataFrame.
        columns: Columns to encode (default: all object/category columns).
        drop_first: Whether to drop the first category to avoid collinearity.

    Returns:
        Tuple of (encoded DataFrame, mapping dict).
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not columns:
        return df.copy(), {}

    encoded = pd.get_dummies(df, columns=columns, drop_first=drop_first)

    mapping: Dict[str, List[str]] = {}
    for col in columns:
        generated = [c for c in encoded.columns if c.startswith(f"{col}_")]
        mapping[col] = generated

    return encoded, mapping


def prepare_features_and_target(
    df: pd.DataFrame,
    target: str,
    sensitive_features: List[str],
    exclude_sensitive: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Separate a DataFrame into feature matrix X, target vector y,
    and sensitive feature matrix S.

    Args:
        df: Full dataset.
        target: Name of the target column.
        sensitive_features: Names of sensitive/protected attribute columns.
        exclude_sensitive: If True, exclude sensitive features from X.

    Returns:
        Tuple of (X, y, S).
    """
    validate_dataframe(df, required_columns=[target] + sensitive_features)

    y = df[target].copy()
    drop_cols = [target]
    if exclude_sensitive:
        drop_cols.extend(sensitive_features)

    X = df.drop(columns=drop_cols, errors="ignore").copy()
    S = df[sensitive_features].copy()

    return X, y, S


def compute_group_proportions(
    series: pd.Series,
) -> Dict[str, float]:
    """
    Compute the proportion of each unique value in a Series.

    Args:
        series: A pandas Series.

    Returns:
        Dict mapping each value to its proportion.
    """
    counts = series.value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def compute_group_outcome_rates(
    group_series: pd.Series,
    target_series: pd.Series,
    positive_label: Any = 1,
) -> Dict[str, float]:
    """
    Compute the positive outcome rate for each group.

    Args:
        group_series: Series of group labels.
        target_series: Series of target values.
        positive_label: The value considered as positive outcome.

    Returns:
        Dict mapping each group to its positive outcome rate.
    """
    combined = pd.DataFrame({"group": group_series, "target": target_series})
    rates = combined.groupby("group")["target"].apply(
        lambda x: (x == positive_label).mean()
    )
    return {str(k): float(v) for k, v in rates.items()}


# ---------------------------------------------------------------------------
# Numeric Helpers
# ---------------------------------------------------------------------------

def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Safely divide two numbers, returning a default if denominator is zero.

    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Value to return if denominator is zero or near-zero.

    Returns:
        The quotient or the default value.
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def clip_value(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clip a value to [low, high] range."""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Configuration Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Classifier thresholds
    "representation_imbalance_threshold": 0.2,
    "outcome_disparity_threshold": 0.1,
    "fairness_violation_threshold": 0.05,
    "proxy_correlation_threshold": 0.3,
    "intersectional_threshold": 0.15,
    "label_noise_threshold": 0.1,

    # Ranker weights
    "alpha": 0.6,  # Weight for accuracy
    "beta": 0.4,   # Weight for fairness (unfairness penalty)

    # Training
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,

    # SMOTE
    "smote_k_neighbors": 5,

    # Threshold optimization
    "threshold_search_steps": 50,

    # LLM
    "gemini_model": "gemini-2.5-flash",
    "gemini_enabled": True,
    "gemini_temperature": 0.3,
    "gemini_max_tokens": 4096,
}


def get_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a configuration dictionary with defaults, optionally overridden.

    Args:
        overrides: Key-value pairs to override defaults.

    Returns:
        Merged configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
