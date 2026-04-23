"""
AEGIS — Unified Bias Detector
================================

Phase 1 of the AEGIS pipeline.  Analyses a dataset for multiple types
of bias and produces a structured report compatible with the mitigation
engine (Phase 2).

This module replaces both:
    - ``src/phase0/inspector.py`` (distribution, outcome, fairness)
    - Inline ``detect_bias()`` in the old ``run_pipeline.py``

Detected bias dimensions:
    - Distribution bias (group representation imbalance)
    - Outcome bias (positive-outcome rate disparity)
    - Fairness metrics (demographic parity, disparate impact)
    - Proxy bias (non-sensitive features correlated with sensitive ones)
    - Intersectional bias (bias at intersections of protected attributes)

Output contract (matches Phase 2 input)::

    {
        "distribution_bias": {feature: {group_proportions, imbalance_ratio}},
        "outcome_bias": {feature: {outcome_rates, disparity}},
        "fairness_metrics": {feature: {demographic_parity_difference, disparate_impact_ratio}},
        "advanced_bias": {"proxy_bias": {}, "intersectional_bias": {}, "label_bias": {}},
        "insights": [str, ...]
    }
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_bias(
    df: pd.DataFrame,
    target: str,
    sensitive_features: List[str],
) -> Dict[str, Any]:
    """
    Perform comprehensive bias detection on a dataset.

    Args:
        df: The dataset to analyse (may contain encoded or raw values).
        target: Target column name for outcome analysis.
        sensitive_features: List of sensitive/protected attribute column names.

    Returns:
        Structured bias report dict with keys:
            - distribution_bias
            - outcome_bias
            - fairness_metrics
            - advanced_bias
            - insights
    """
    logger.info(f"Running bias detection on {len(df)} rows...")
    logger.info(f"  Target: '{target}'")
    logger.info(f"  Sensitive features: {sensitive_features}")

    report: Dict[str, Any] = {
        "distribution_bias": {},
        "outcome_bias": {},
        "fairness_metrics": {},
        "advanced_bias": {
            "proxy_bias": {},
            "intersectional_bias": {},
            "label_bias": {},
        },
        "insights": [],
    }

    # --- Per-feature analysis ---
    for feat in sensitive_features:
        if feat not in df.columns:
            logger.warning(f"Sensitive feature '{feat}' not in dataset — skipping.")
            continue

        _analyse_distribution(df, feat, report)
        _analyse_outcome(df, feat, target, report)
        _analyse_fairness(df, feat, target, report)

    # --- Cross-feature analysis ---
    _analyse_proxy_bias(df, target, sensitive_features, report)
    _analyse_intersectional_bias(df, target, sensitive_features, report)

    logger.info(f"Bias detection complete — {len(report['insights'])} insights found")
    return report


# ---------------------------------------------------------------------------
# Internal Analysis Functions
# ---------------------------------------------------------------------------

def _analyse_distribution(
    df: pd.DataFrame,
    feature: str,
    report: Dict[str, Any],
) -> None:
    """Analyse group representation distribution."""
    props = df[feature].value_counts(normalize=True).to_dict()
    imbalance = max(props.values()) / max(min(props.values()), 1e-10)

    report["distribution_bias"][feature] = {
        "group_proportions": {str(k): round(float(v), 4) for k, v in props.items()},
        "imbalance_ratio": round(imbalance, 4),
    }

    if imbalance > 1.5:
        report["insights"].append(
            f"Representation imbalance in '{feature}': ratio = {imbalance:.2f}"
        )


def _analyse_outcome(
    df: pd.DataFrame,
    feature: str,
    target: str,
    report: Dict[str, Any],
) -> None:
    """Analyse outcome disparity across groups."""
    if target not in df.columns:
        return

    outcome_rates = df.groupby(feature)[target].mean().to_dict()
    disparity = max(outcome_rates.values()) - min(outcome_rates.values())

    report["outcome_bias"][feature] = {
        "outcome_rates": {str(k): round(float(v), 4) for k, v in outcome_rates.items()},
        "disparity": round(disparity, 4),
    }

    if disparity > 0.05:
        report["insights"].append(
            f"Outcome disparity in '{feature}': gap = {disparity:.2%}"
        )


def _analyse_fairness(
    df: pd.DataFrame,
    feature: str,
    target: str,
    report: Dict[str, Any],
) -> None:
    """Compute standard fairness metrics."""
    if target not in df.columns:
        return

    outcome_rates = df.groupby(feature)[target].mean().to_dict()
    rates = list(outcome_rates.values())

    if len(rates) < 2:
        return

    dp_diff = max(rates) - min(rates)
    di_ratio = min(rates) / max(max(rates), 1e-10)

    report["fairness_metrics"][feature] = {
        "demographic_parity_difference": round(dp_diff, 4),
        "disparate_impact_ratio": round(di_ratio, 4),
    }


def _analyse_proxy_bias(
    df: pd.DataFrame,
    target: str,
    sensitive_features: List[str],
    report: Dict[str, Any],
) -> None:
    """Detect proxy features correlated with sensitive attributes."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    for feat in sensitive_features:
        if feat not in df.columns:
            continue

        feat_encoded = pd.factorize(df[feat])[0]

        for col in numeric_cols:
            if col in sensitive_features:
                continue
            try:
                corr = abs(np.corrcoef(feat_encoded, df[col].values)[0, 1])
                if corr > 0.5:
                    report["advanced_bias"]["proxy_bias"][col] = {
                        "correlation": round(corr, 4),
                        "correlated_with": feat,
                    }
                    report["insights"].append(
                        f"Proxy bias: '{col}' correlates with '{feat}' (r={corr:.3f})"
                    )
            except Exception:
                continue


def _analyse_intersectional_bias(
    df: pd.DataFrame,
    target: str,
    sensitive_features: List[str],
    report: Dict[str, Any],
) -> None:
    """Detect bias at intersections of multiple protected attributes."""
    if len(sensitive_features) < 2 or target not in df.columns:
        return

    try:
        combo_col = (
            df[sensitive_features[0]].astype(str) + "_" +
            df[sensitive_features[1]].astype(str)
        )
        combo_rates = df.groupby(combo_col)[target].mean()

        max_disparity = float(combo_rates.max() - combo_rates.min())
        if max_disparity > 0.15:
            report["advanced_bias"]["intersectional_bias"] = {
                "features": sensitive_features[:2],
                "max_disparity": round(max_disparity, 4),
                "group_rates": {
                    str(k): round(float(v), 4) for k, v in combo_rates.items()
                },
            }
            report["insights"].append(
                f"Intersectional bias detected across "
                f"{sensitive_features[0]} x {sensitive_features[1]}"
            )
    except Exception:
        pass
