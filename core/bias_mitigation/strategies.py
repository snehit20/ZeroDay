"""
AEGIS -- Mitigation Strategies
================================

Implements all bias mitigation techniques used by the engine.

Supported techniques:
    1. **Reweighting**           -- P(y) / P(y|group) sample weights
    2. **Resampling**            -- random over/undersampling
    3. **SMOTE**                 -- synthetic minority oversampling
    4. **Disparate Impact Remover** -- feature distribution repair
    5. **Threshold Optimization** -- group-specific decision thresholds
    6. **Fairlearn Reduction**   -- constrained optimisation via
                                    ExponentiatedGradient

Each technique returns a ``MitigationResult`` dict containing the
transformed data and/or model artefacts needed by the Trainer.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from .utils import (
    check_dependency,
    compute_group_outcome_rates,
    compute_group_proportions,
    get_config,
    get_logger,
    safe_divide,
    safe_import,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

MitigationResult = Dict[str, Any]
# Common keys:
#   "X"             : transformed feature matrix (pd.DataFrame)
#   "y"             : target vector (pd.Series)
#   "sample_weight" : per-sample weights (np.ndarray or None)
#   "model"         : a pre-fitted model (for post-processing strategies)
#   "thresholds"    : group-specific thresholds dict
#   "metadata"      : extra info about the transformation


# =====================================================================
# 1.  REWEIGHTING
# =====================================================================

def apply_reweighting(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    config: Optional[Dict[str, Any]] = None,
) -> MitigationResult:
    """
    Compute sample weights using:  ``weight = P(y) / P(y | group)``

    Each sample receives a weight that compensates for the over- or
    under-representation of its (group, label) combination.

    Args:
        X: Feature matrix.
        y: Target vector (binary 0/1).
        sensitive: Series of group labels for the primary sensitive feature.
        config: Optional configuration overrides.

    Returns:
        MitigationResult with ``sample_weight`` populated.
    """
    logger.info("Applying reweighting strategy ...")
    cfg = get_config(config)

    y_vals = y.values if isinstance(y, pd.Series) else np.asarray(y)
    s_vals = sensitive.values if isinstance(sensitive, pd.Series) else np.asarray(sensitive)

    n = len(y_vals)
    unique_labels = np.unique(y_vals)
    unique_groups = np.unique(s_vals)

    # P(y) -- marginal label probability
    p_y = {label: np.mean(y_vals == label) for label in unique_labels}

    # P(y | group) -- conditional label probability per group
    weights = np.ones(n, dtype=np.float64)

    for label in unique_labels:
        for group in unique_groups:
            mask = (y_vals == label) & (s_vals == group)
            group_mask = s_vals == group
            p_y_given_group = safe_divide(
                float(mask.sum()), float(group_mask.sum()), default=1.0
            )
            w = safe_divide(p_y[label], p_y_given_group, default=1.0)
            weights[mask] = w

    # Normalise so mean weight ≈ 1
    weights = weights / weights.mean()

    logger.info(
        f"Reweighting complete -- weight range: [{weights.min():.3f}, {weights.max():.3f}]"
    )

    return {
        "X": X.copy(),
        "y": y.copy(),
        "sample_weight": weights,
        "model": None,
        "thresholds": None,
        "metadata": {
            "technique": "reweighting",
            "num_groups": len(unique_groups),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
        },
    }


# =====================================================================
# 2.  RESAMPLING
# =====================================================================

def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    method: str = "oversample",
    config: Optional[Dict[str, Any]] = None,
) -> MitigationResult:
    """
    Balance group representation via random over- or under-sampling.

    Args:
        X: Feature matrix.
        y: Target vector.
        sensitive: Group labels.
        method: One of ``"oversample"``, ``"undersample"``, ``"hybrid"``.
        config: Optional configuration overrides.

    Returns:
        MitigationResult with resampled X and y.
    """
    logger.info(f"Applying resampling ({method}) ...")

    combined = pd.DataFrame(X).copy()
    combined["__target__"] = y.values
    combined["__group__"] = sensitive.values

    groups = combined["__group__"].unique()
    group_counts = combined["__group__"].value_counts()
    target_count = int(group_counts.median()) if method == "hybrid" else (
        int(group_counts.max()) if method == "oversample" else int(group_counts.min())
    )

    resampled_frames: List[pd.DataFrame] = []
    rng = np.random.RandomState(get_config(config)["random_state"])

    for group in groups:
        group_df = combined[combined["__group__"] == group]
        n = len(group_df)

        if n < target_count:
            # Oversample
            extra = group_df.sample(n=target_count - n, replace=True, random_state=rng)
            resampled_frames.append(pd.concat([group_df, extra], ignore_index=True))
        elif n > target_count:
            # Undersample
            resampled_frames.append(
                group_df.sample(n=target_count, replace=False, random_state=rng)
            )
        else:
            resampled_frames.append(group_df)

    result_df = pd.concat(resampled_frames, ignore_index=True)
    # Shuffle
    result_df = result_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    y_out = result_df.pop("__target__")
    result_df.pop("__group__")

    logger.info(
        f"Resampling complete -- {len(X)} -> {len(result_df)} samples"
    )

    return {
        "X": result_df,
        "y": y_out,
        "sample_weight": None,
        "model": None,
        "thresholds": None,
        "metadata": {
            "technique": "resampling",
            "method": method,
            "original_size": len(X),
            "resampled_size": len(result_df),
        },
    }


# =====================================================================
# 3.  SMOTE
# =====================================================================

def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    config: Optional[Dict[str, Any]] = None,
) -> MitigationResult:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique).

    Uses ``imbalanced-learn`` if available, otherwise falls back to
    basic random oversampling.

    Args:
        X: Feature matrix (numeric features only).
        y: Target vector.
        sensitive: Group labels.
        config: Optional configuration overrides.

    Returns:
        MitigationResult with oversampled X and y.
    """
    logger.info("Applying SMOTE ...")
    cfg = get_config(config)

    # Encode categoricals for SMOTE
    X_numeric = X.copy()
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in X_numeric.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
        label_encoders[col] = le

    imblearn = safe_import("imblearn.over_sampling")

    if imblearn is not None:
        try:
            smote = imblearn.SMOTE(
                k_neighbors=min(cfg["smote_k_neighbors"], len(y) - 1),
                random_state=cfg["random_state"],
            )
            X_res, y_res = smote.fit_resample(X_numeric, y)
            X_res = pd.DataFrame(X_res, columns=X_numeric.columns)
            y_res = pd.Series(y_res, name=y.name)

            logger.info(
                f"SMOTE (imbalanced-learn) complete -- {len(X)} -> {len(X_res)} samples"
            )
            return {
                "X": X_res,
                "y": y_res,
                "sample_weight": None,
                "model": None,
                "thresholds": None,
                "metadata": {
                    "technique": "smote",
                    "backend": "imbalanced-learn",
                    "original_size": len(X),
                    "resampled_size": len(X_res),
                },
            }
        except Exception as e:
            logger.warning(f"SMOTE via imbalanced-learn failed: {e}. Falling back.")

    # Fallback: random oversampling of minority class
    logger.info("SMOTE fallback -> random oversampling of minority class")
    return apply_resampling(X, y, sensitive, method="oversample", config=config)


# =====================================================================
# 4.  DISPARATE IMPACT REMOVER
# =====================================================================

def apply_disparate_impact_remover(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    repair_level: float = 1.0,
    config: Optional[Dict[str, Any]] = None,
) -> MitigationResult:
    """
    Remove disparate impact by repairing feature distributions so that
    they are independent of the sensitive attribute.

    Uses AIF360 ``DisparateImpactRemover`` if available; otherwise applies
    a quantile-based feature normalisation approach.

    Args:
        X: Feature matrix.
        y: Target vector.
        sensitive: Group labels.
        repair_level: Degree of repair (0 = no repair, 1 = full repair).
        config: Optional configuration overrides.

    Returns:
        MitigationResult with transformed X.
    """
    logger.info(f"Applying Disparate Impact Remover (repair={repair_level}) ...")

    # --- Try AIF360 ---
    aif360 = safe_import("aif360.algorithms.preprocessing")
    if aif360 is not None:
        try:
            return _dir_aif360(X, y, sensitive, repair_level, config)
        except Exception as e:
            logger.warning(f"AIF360 DIR failed: {e}. Using quantile fallback.")

    # --- Fallback: quantile-based repair ---
    return _dir_quantile_repair(X, y, sensitive, repair_level, config)


def _dir_aif360(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    repair_level: float,
    config: Optional[Dict[str, Any]],
) -> MitigationResult:
    """AIF360-based disparate impact removal."""
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR

    # Build AIF360 dataset
    sens_name = sensitive.name or "sensitive"
    df_combined = X.copy()
    df_combined[sens_name] = LabelEncoder().fit_transform(sensitive.astype(str))
    target_name = y.name or "target"
    df_combined[target_name] = y.values

    dataset = BinaryLabelDataset(
        df=df_combined,
        label_names=[target_name],
        protected_attribute_names=[sens_name],
    )

    remover = DIR(repair_level=repair_level)
    repaired = remover.fit_transform(dataset)
    repaired_df = repaired.convert_to_dataframe()[0]

    X_out = repaired_df.drop(columns=[target_name, sens_name], errors="ignore")
    y_out = repaired_df[target_name]

    logger.info("DIR (AIF360) complete.")
    return {
        "X": X_out.reset_index(drop=True),
        "y": pd.Series(y_out.values, name=y.name).reset_index(drop=True),
        "sample_weight": None,
        "model": None,
        "thresholds": None,
        "metadata": {
            "technique": "disparate_impact_remover",
            "backend": "aif360",
            "repair_level": repair_level,
        },
    }


def _dir_quantile_repair(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    repair_level: float,
    config: Optional[Dict[str, Any]],
) -> MitigationResult:
    """
    Quantile-based feature repair.

    For each numeric feature, the per-group distribution is shifted toward
    the median distribution across all groups.  ``repair_level`` controls
    how far the shift goes (0 = none, 1 = full convergence to median).
    """
    X_repaired = X.copy()
    numeric_cols = X_repaired.select_dtypes(include=[np.number]).columns.tolist()

    groups = sensitive.unique()

    for col in numeric_cols:
        # Compute per-group quantile functions
        overall_median = X_repaired[col].median()
        overall_std = X_repaired[col].std()

        if overall_std < 1e-10:
            continue

        for group in groups:
            mask = sensitive == group
            group_vals = X_repaired.loc[mask, col]
            group_median = group_vals.median()
            group_std = group_vals.std()

            if group_std < 1e-10:
                continue

            # Shift: move group distribution toward overall distribution
            shift = (overall_median - group_median) * repair_level
            scale = 1.0 + (safe_divide(overall_std, group_std, 1.0) - 1.0) * repair_level

            X_repaired.loc[mask, col] = (
                (group_vals - group_median) * scale + group_median + shift
            )

    logger.info("DIR (quantile repair fallback) complete.")
    return {
        "X": X_repaired.reset_index(drop=True),
        "y": y.copy().reset_index(drop=True),
        "sample_weight": None,
        "model": None,
        "thresholds": None,
        "metadata": {
            "technique": "disparate_impact_remover",
            "backend": "quantile_repair",
            "repair_level": repair_level,
            "features_repaired": numeric_cols,
        },
    }


# =====================================================================
# 5.  THRESHOLD OPTIMIZATION
# =====================================================================

def apply_threshold_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    estimator: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> MitigationResult:
    """
    Find group-specific classification thresholds that equalise positive
    outcome rates across groups (demographic parity).

    If no pre-trained estimator is provided, a Logistic Regression model
    is fitted first.

    Args:
        X: Feature matrix (numeric).
        y: Target vector.
        sensitive: Group labels.
        estimator: Optional pre-trained model with ``predict_proba``.
        config: Optional configuration overrides.

    Returns:
        MitigationResult with optimised thresholds and the fitted model.
    """
    logger.info("Applying threshold optimization ...")
    cfg = get_config(config)

    # Encode categoricals
    X_numeric = X.copy()
    for col in X_numeric.select_dtypes(include=["object", "category"]).columns:
        X_numeric[col] = LabelEncoder().fit_transform(X_numeric[col].astype(str))

    # Fit model if needed
    if estimator is None:
        estimator = LogisticRegression(
            max_iter=1000,
            random_state=cfg["random_state"],
            solver="lbfgs",
        )
        estimator.fit(X_numeric, y)

    # Predict probabilities
    probas = estimator.predict_proba(X_numeric)[:, 1]
    groups = sensitive.unique()
    steps = cfg["threshold_search_steps"]

    # Overall positive rate
    overall_rate = float(y.mean())

    # Find per-group threshold that achieves closest to overall_rate
    thresholds: Dict[str, float] = {}

    for group in groups:
        mask = sensitive == group
        group_probas = probas[mask]
        best_thresh = 0.5
        best_diff = float("inf")

        for t in np.linspace(0.0, 1.0, steps):
            predicted_rate = float((group_probas >= t).mean())
            diff = abs(predicted_rate - overall_rate)
            if diff < best_diff:
                best_diff = diff
                best_thresh = float(t)

        thresholds[str(group)] = round(best_thresh, 4)

    logger.info(f"Optimised thresholds: {thresholds}")

    return {
        "X": X.copy(),
        "y": y.copy(),
        "sample_weight": None,
        "model": estimator,
        "thresholds": thresholds,
        "metadata": {
            "technique": "threshold_optimization",
            "thresholds": thresholds,
            "overall_positive_rate": overall_rate,
        },
    }


# =====================================================================
# 6.  FAIRLEARN REDUCTION
# =====================================================================

def apply_fairlearn_reduction(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    constraint: str = "demographic_parity",
    config: Optional[Dict[str, Any]] = None,
) -> MitigationResult:
    """
    Train a model under fairness constraints using Fairlearn's
    Exponentiated Gradient reduction.

    Args:
        X: Feature matrix.
        y: Target vector.
        sensitive: Group labels.
        constraint: Fairness constraint -- one of
            ``"demographic_parity"``, ``"equalized_odds"``,
            ``"true_positive_rate_parity"``.
        config: Optional configuration overrides.

    Returns:
        MitigationResult with the constrained model.
    """
    logger.info(f"Applying Fairlearn reduction (constraint={constraint}) ...")
    cfg = get_config(config)

    # Encode categoricals
    X_numeric = X.copy()
    for col in X_numeric.select_dtypes(include=["object", "category"]).columns:
        X_numeric[col] = LabelEncoder().fit_transform(X_numeric[col].astype(str))

    fairlearn = safe_import("fairlearn.reductions")

    if fairlearn is not None:
        try:
            # Select constraint
            constraint_map = {
                "demographic_parity": fairlearn.DemographicParity,
                "equalized_odds": fairlearn.EqualizedOdds,
                "true_positive_rate_parity": fairlearn.TruePositiveRateParity,
            }
            constraint_cls = constraint_map.get(constraint, fairlearn.DemographicParity)
            constraint_obj = constraint_cls()

            base_estimator = LogisticRegression(
                max_iter=1000,
                random_state=cfg["random_state"],
                solver="lbfgs",
            )

            mitigator = fairlearn.ExponentiatedGradient(
                estimator=base_estimator,
                constraints=constraint_obj,
            )
            mitigator.fit(X_numeric, y, sensitive_features=sensitive)

            logger.info("Fairlearn reduction complete.")
            return {
                "X": X.copy(),
                "y": y.copy(),
                "sample_weight": None,
                "model": mitigator,
                "thresholds": None,
                "metadata": {
                    "technique": "fairlearn_reduction",
                    "backend": "fairlearn",
                    "constraint": constraint,
                },
            }
        except Exception as e:
            logger.warning(f"Fairlearn reduction failed: {e}. Using reweighting fallback.")

    # Fallback: use reweighting as a simpler fairness-aware approach
    logger.info("Fairlearn not available -- falling back to reweighting.")
    result = apply_reweighting(X, y, sensitive, config=config)
    result["metadata"]["technique"] = "fairlearn_reduction_fallback"
    return result


# =====================================================================
# STRATEGY DISPATCHER
# =====================================================================

STRATEGY_FUNCTIONS = {
    "reweighting": apply_reweighting,
    "resampling": apply_resampling,
    "smote": apply_smote,
    "disparate_impact_remover": apply_disparate_impact_remover,
    "threshold_optimization": apply_threshold_optimization,
    "fairlearn_reduction": apply_fairlearn_reduction,
}


def execute_strategy(
    strategy_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> MitigationResult:
    """
    Execute a named mitigation strategy.

    For combined strategies (e.g. ``"disparate_impact_remover + reweighting"``),
    the strategies are applied sequentially -- each step receives the output
    of the previous one.

    Args:
        strategy_name: Strategy name or combined pipeline string.
        X: Feature matrix.
        y: Target vector.
        sensitive: Group labels.
        config: Optional configuration overrides.
        **kwargs: Extra keyword arguments passed to the strategy function.

    Returns:
        MitigationResult from the final strategy in the pipeline.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    # Handle combined strategies
    steps = [s.strip() for s in strategy_name.split("+")]
    # Normalise aliases
    ALIASES = {
        "dir": "disparate_impact_remover",
        "di_remover": "disparate_impact_remover",
        "fl_reduction": "fairlearn_reduction",
        "threshold_opt": "threshold_optimization",
    }

    current_X = X.copy()
    current_y = y.copy()
    current_weight = None
    current_model = None
    current_thresholds = None
    all_metadata: List[Dict[str, Any]] = []

    for step in steps:
        step_name = ALIASES.get(step, step)

        if step_name == "baseline":
            # No transformation
            all_metadata.append({"technique": "baseline"})
            continue

        func = STRATEGY_FUNCTIONS.get(step_name)
        if func is None:
            raise ValueError(
                f"Unknown strategy '{step_name}'.  "
                f"Available: {list(STRATEGY_FUNCTIONS.keys())}"
            )

        result = func(current_X, current_y, sensitive, config=config, **kwargs)
        current_X = result["X"]
        current_y = result["y"]

        if result.get("sample_weight") is not None:
            current_weight = result["sample_weight"]
        if result.get("model") is not None:
            current_model = result["model"]
        if result.get("thresholds") is not None:
            current_thresholds = result["thresholds"]

        all_metadata.append(result.get("metadata", {}))

    return {
        "X": current_X,
        "y": current_y,
        "sample_weight": current_weight,
        "model": current_model,
        "thresholds": current_thresholds,
        "metadata": {
            "pipeline": strategy_name,
            "steps": all_metadata,
        },
    }
