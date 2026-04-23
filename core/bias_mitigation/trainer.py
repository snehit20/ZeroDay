"""
AEGIS -- Model Trainer
=======================

Trains separate models for every candidate mitigation pipeline.

Supported model types:
    - Logistic Regression  (baseline, always available)
    - Random Forest
    - XGBoost (optional -- graceful fallback if not installed)

Each pipeline receives its own training run using the transformed data
(and optional sample weights) produced by the strategies module.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .strategies import execute_strategy, MitigationResult
from .utils import get_config, get_logger, safe_import

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

TrainedModel = Dict[str, Any]
# {
#     "pipeline": str,
#     "model_type": str,
#     "model": <fitted sklearn estimator>,
#     "X_train": pd.DataFrame,
#     "X_test": pd.DataFrame,
#     "y_train": pd.Series,
#     "y_test": pd.Series,
#     "sensitive_train": pd.Series,
#     "sensitive_test": pd.Series,
#     "predictions": np.ndarray,
#     "probabilities": np.ndarray,
#     "mitigation_result": MitigationResult,
# }

TrainingOutput = Dict[str, TrainedModel]


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def _get_model_instance(
    model_type: str,
    config: Dict[str, Any],
    supports_sample_weight: bool = True,
) -> Any:
    """
    Instantiate a model by name.

    Args:
        model_type: One of ``"logistic_regression"``, ``"random_forest"``,
            ``"xgboost"``.
        config: Configuration dict (for random_state, etc.).
        supports_sample_weight: Hint -- not all wrappers need this.

    Returns:
        An unfitted sklearn-compatible estimator.

    Raises:
        ValueError: If the model type is unknown and no fallback exists.
    """
    rs = config.get("random_state", 42)

    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=rs, solver="lbfgs")

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1
        )

    if model_type == "xgboost":
        xgb = safe_import("xgboost")
        if xgb is not None:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=rs,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:
            logger.warning("XGBoost not available -- falling back to Random Forest.")
            return RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1
            )

    raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Encoding Helper
# ---------------------------------------------------------------------------

def _encode_for_training(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical columns for sklearn models."""
    X_enc = X.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in X_enc.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encoders[col] = le
    return X_enc, encoders


# ---------------------------------------------------------------------------
# Single Pipeline Trainer
# ---------------------------------------------------------------------------

def _train_single_pipeline(
    pipeline_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    model_type: str = "logistic_regression",
    config: Optional[Dict[str, Any]] = None,
) -> TrainedModel:
    """
    Execute a single mitigation pipeline and train a model on the result.

    Args:
        pipeline_name: Strategy pipeline name (e.g. ``"reweighting"``).
        X: Original feature matrix.
        y: Original target vector.
        sensitive: Original sensitive feature Series.
        model_type: Model type to train.
        config: Configuration overrides.

    Returns:
        TrainedModel dict with all artefacts.
    """
    cfg = get_config(config)
    logger.info(f"Training pipeline '{pipeline_name}' with model '{model_type}' ...")

    # --- Apply mitigation ---
    if pipeline_name == "baseline":
        mitigation_result: MitigationResult = {
            "X": X.copy(),
            "y": y.copy(),
            "sample_weight": None,
            "model": None,
            "thresholds": None,
            "metadata": {"technique": "baseline"},
        }
    else:
        mitigation_result = execute_strategy(
            pipeline_name, X, y, sensitive, config=cfg
        )

    mit_X = mitigation_result["X"]
    mit_y = mitigation_result["y"]
    sample_weight = mitigation_result.get("sample_weight")
    pre_fitted_model = mitigation_result.get("model")
    thresholds = mitigation_result.get("thresholds")

    # --- Encode ---
    mit_X_enc, encoders = _encode_for_training(mit_X)

    # Reset indices to ensure alignment across X, y, sensitive, weights
    mit_X_enc = mit_X_enc.reset_index(drop=True)
    mit_y = mit_y.reset_index(drop=True)

    # Handle sensitive alignment (resampling may have changed length)
    if len(mit_X_enc) != len(sensitive):
        sensitive_aligned = pd.Series(
            ["unknown"] * len(mit_X_enc), name=sensitive.name
        )
    else:
        sensitive_aligned = sensitive.reset_index(drop=True)

    # Wrap sample_weight as a Series so it participates in train_test_split
    if sample_weight is not None:
        sw_series = pd.Series(sample_weight).reset_index(drop=True)
    else:
        sw_series = None

    # --- Train/test split ---
    if sw_series is not None:
        X_train, X_test, y_train, y_test, s_train, s_test, sw_train, sw_test = (
            train_test_split(
                mit_X_enc,
                mit_y,
                sensitive_aligned,
                sw_series,
                test_size=cfg["test_size"],
                random_state=cfg["random_state"],
                stratify=mit_y if len(mit_y.unique()) > 1 else None,
            )
        )
    else:
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            mit_X_enc,
            mit_y,
            sensitive_aligned,
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
            stratify=mit_y if len(mit_y.unique()) > 1 else None,
        )
        sw_train = None

    # --- Fit model ---
    if pre_fitted_model is not None:
        # Use the model already fitted by the strategy (e.g. Fairlearn, threshold opt)
        model = pre_fitted_model
    else:
        model = _get_model_instance(model_type, cfg)
        fit_kwargs: Dict[str, Any] = {}
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train.values
        model.fit(X_train, y_train, **fit_kwargs)

    # --- Predict ---
    if thresholds is not None:
        # Apply group-specific thresholds
        probabilities = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_test).astype(float)
        )
        predictions = np.zeros(len(X_test), dtype=int)
        for group, thresh in thresholds.items():
            mask = s_test == group
            if mask.any():
                predictions[mask.values] = (probabilities[mask.values] >= thresh).astype(int)
        # Handle unknown groups with default 0.5
        unknown_mask = ~s_test.isin(thresholds.keys())
        if unknown_mask.any():
            predictions[unknown_mask.values] = (
                probabilities[unknown_mask.values] >= 0.5
            ).astype(int)
    else:
        predictions = model.predict(X_test)
        probabilities = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else predictions.astype(float)
        )

    logger.info(f"Pipeline '{pipeline_name}' training complete.")

    return {
        "pipeline": pipeline_name,
        "model_type": model_type,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_train": s_train,
        "sensitive_test": s_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "mitigation_result": mitigation_result,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_models(
    candidate_pipelines: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    model_type: str = "logistic_regression",
    config: Optional[Dict[str, Any]] = None,
) -> TrainingOutput:
    """
    Train a model for every candidate mitigation pipeline.

    Args:
        candidate_pipelines: List of pipeline names from the generator.
        X: Original feature matrix.
        y: Original target vector.
        sensitive: Primary sensitive feature Series.
        model_type: Model type to use for all pipelines.
        config: Configuration overrides.

    Returns:
        Dictionary mapping pipeline names to TrainedModel dicts.
    """
    cfg = get_config(config)
    logger.info(
        f"Training {len(candidate_pipelines)} pipelines with '{model_type}' ..."
    )

    results: TrainingOutput = {}

    for pipeline in candidate_pipelines:
        try:
            trained = _train_single_pipeline(
                pipeline_name=pipeline,
                X=X,
                y=y,
                sensitive=sensitive,
                model_type=model_type,
                config=cfg,
            )
            results[pipeline] = trained
        except Exception as e:
            logger.error(f"Pipeline '{pipeline}' failed: {e}", exc_info=True)
            continue

    logger.info(
        f"Successfully trained {len(results)}/{len(candidate_pipelines)} pipelines."
    )
    return results
