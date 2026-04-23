"""
AEGIS — Dataset Comparator
=============================

Phase 3 of the AEGIS pipeline.  Compares the baseline (original) dataset
against the debiased dataset produced by the mitigation engine.

Analysis performed:
    - Representation distribution shifts per sensitive feature
    - Outcome rate changes before/after
    - Fairness metric deltas (demographic parity, disparate impact)
    - Statistical distribution comparison (KS test)
    - Summary statistics comparison
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_datasets(
    baseline_dataset: pd.DataFrame,
    debiased_dataset: pd.DataFrame,
    target: str,
    sensitive_features: List[str],
    bias_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare baseline vs debiased datasets across multiple dimensions.

    Args:
        baseline_dataset: Original dataset before mitigation.
        debiased_dataset: Dataset after mitigation.
        target: Target column name.
        sensitive_features: List of sensitive feature column names.
        bias_report: Optional bias report from Phase 1 (for additional context).

    Returns:
        Comparison report dict with keys:
            - baseline_stats
            - debiased_stats
            - representation_shift
            - outcome_rate_change
            - fairness_deltas
            - statistical_tests
            - summary
    """
    logger.info("Running dataset comparison (Phase 3)...")
    logger.info(f"  Baseline shape: {baseline_dataset.shape}")
    logger.info(f"  Debiased shape: {debiased_dataset.shape}")

    result = {
        "baseline_stats": _compute_stats(baseline_dataset, target),
        "debiased_stats": _compute_stats(debiased_dataset, target),
        "representation_shift": {},
        "outcome_rate_change": {},
        "fairness_deltas": {},
        "statistical_tests": {},
        "summary": {},
    }

    for feat in sensitive_features:
        # Representation shift
        if feat in baseline_dataset.columns and feat in debiased_dataset.columns:
            result["representation_shift"][feat] = _compare_representation(
                baseline_dataset, debiased_dataset, feat
            )

        # Outcome rate change
        if (feat in baseline_dataset.columns and feat in debiased_dataset.columns
                and target in baseline_dataset.columns and target in debiased_dataset.columns):
            result["outcome_rate_change"][feat] = _compare_outcome_rates(
                baseline_dataset, debiased_dataset, feat, target
            )

            # Fairness metric deltas
            result["fairness_deltas"][feat] = _compare_fairness(
                baseline_dataset, debiased_dataset, feat, target
            )

    # Statistical tests on numeric columns
    result["statistical_tests"] = _statistical_comparison(
        baseline_dataset, debiased_dataset
    )

    # Summary
    result["summary"] = _generate_summary(result, sensitive_features)

    logger.info("Dataset comparison complete.")
    return result


# ---------------------------------------------------------------------------
# Internal Analysis Functions
# ---------------------------------------------------------------------------

def _compute_stats(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Compute summary statistics for a dataset."""
    stats = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": int(df.isnull().sum().sum()),
    }

    if target in df.columns:
        target_vals = df[target]
        stats["target_distribution"] = {
            str(k): int(v) for k, v in target_vals.value_counts().items()
        }
        stats["target_positive_rate"] = round(float(target_vals.mean()), 4)

    return stats


def _compare_representation(
    baseline: pd.DataFrame,
    debiased: pd.DataFrame,
    feature: str,
) -> Dict[str, Any]:
    """Compare group representation before/after."""
    before = baseline[feature].value_counts(normalize=True).to_dict()
    after = debiased[feature].value_counts(normalize=True).to_dict()

    all_groups = set(list(before.keys()) + list(after.keys()))
    shifts = {}

    for group in all_groups:
        g = str(group)
        b = before.get(group, 0.0)
        a = after.get(group, 0.0)
        shifts[g] = {
            "before": round(float(b), 4),
            "after": round(float(a), 4),
            "delta": round(float(a - b), 4),
        }

    # Overall imbalance change
    before_values = list(before.values())
    after_values = list(after.values())

    before_imbalance = (max(before_values) - min(before_values)) if before_values else 0
    after_imbalance = (max(after_values) - min(after_values)) if after_values else 0

    return {
        "groups": shifts,
        "imbalance_before": round(before_imbalance, 4),
        "imbalance_after": round(after_imbalance, 4),
        "imbalance_improved": after_imbalance < before_imbalance,
    }


def _compare_outcome_rates(
    baseline: pd.DataFrame,
    debiased: pd.DataFrame,
    feature: str,
    target: str,
) -> Dict[str, Any]:
    """Compare outcome rates before/after per group."""
    before_rates = baseline.groupby(feature)[target].mean().to_dict()
    after_rates = debiased.groupby(feature)[target].mean().to_dict()

    all_groups = set(list(before_rates.keys()) + list(after_rates.keys()))
    changes = {}

    for group in all_groups:
        g = str(group)
        b = before_rates.get(group, 0.0)
        a = after_rates.get(group, 0.0)
        changes[g] = {
            "before": round(float(b), 4),
            "after": round(float(a), 4),
            "delta": round(float(a - b), 4),
        }

    # Disparity change
    before_vals = list(before_rates.values())
    after_vals = list(after_rates.values())

    before_disparity = (max(before_vals) - min(before_vals)) if len(before_vals) >= 2 else 0
    after_disparity = (max(after_vals) - min(after_vals)) if len(after_vals) >= 2 else 0

    return {
        "groups": changes,
        "disparity_before": round(before_disparity, 4),
        "disparity_after": round(after_disparity, 4),
        "disparity_improved": after_disparity < before_disparity,
    }


def _compare_fairness(
    baseline: pd.DataFrame,
    debiased: pd.DataFrame,
    feature: str,
    target: str,
) -> Dict[str, Any]:
    """Compare fairness metrics before/after."""
    def _compute_metrics(df, feat, tgt):
        rates = df.groupby(feat)[tgt].mean()
        val_list = list(rates.values)
        if len(val_list) < 2:
            return {"dp_diff": 0.0, "di_ratio": 1.0}
        dp = max(val_list) - min(val_list)
        di = min(val_list) / max(max(val_list), 1e-10)
        return {
            "dp_diff": round(dp, 4),
            "di_ratio": round(di, 4),
        }

    before = _compute_metrics(baseline, feature, target)
    after = _compute_metrics(debiased, feature, target)

    return {
        "before": before,
        "after": after,
        "dp_diff_delta": round(after["dp_diff"] - before["dp_diff"], 4),
        "di_ratio_delta": round(after["di_ratio"] - before["di_ratio"], 4),
        "dp_improved": after["dp_diff"] < before["dp_diff"],
        "di_improved": after["di_ratio"] > before["di_ratio"],
    }


def _statistical_comparison(
    baseline: pd.DataFrame,
    debiased: pd.DataFrame,
) -> Dict[str, Any]:
    """Run statistical tests comparing distributions."""
    results = {}

    # Only compare shared numeric columns
    shared_numeric = [
        col for col in baseline.select_dtypes(include=[np.number]).columns
        if col in debiased.columns
    ]

    for col in shared_numeric[:10]:  # Limit to first 10 to avoid overhead
        try:
            from scipy.stats import ks_2samp
            stat, pval = ks_2samp(
                baseline[col].dropna().values,
                debiased[col].dropna().values,
            )
            results[col] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "significantly_different": pval < 0.05,
            }
        except ImportError:
            # scipy not available — compute basic stats instead
            b_mean = float(baseline[col].mean())
            d_mean = float(debiased[col].mean())
            results[col] = {
                "mean_before": round(b_mean, 4),
                "mean_after": round(d_mean, 4),
                "mean_delta": round(d_mean - b_mean, 4),
            }
        except Exception:
            continue

    return results


def _generate_summary(
    result: Dict[str, Any],
    sensitive_features: List[str],
) -> Dict[str, Any]:
    """Generate a summary of the dataset comparison."""
    improved_features = []
    worsened_features = []

    for feat in sensitive_features:
        fairness = result.get("fairness_deltas", {}).get(feat, {})
        if fairness.get("dp_improved", False):
            improved_features.append(feat)
        elif fairness.get("dp_diff_delta", 0) > 0:
            worsened_features.append(feat)

    num_stat_tests = len(result.get("statistical_tests", {}))
    num_significant = sum(
        1 for v in result.get("statistical_tests", {}).values()
        if v.get("significantly_different", False)
    )

    return {
        "features_improved": improved_features,
        "features_worsened": worsened_features,
        "dataset_size_change": {
            "before": result["baseline_stats"]["shape"][0],
            "after": result["debiased_stats"]["shape"][0],
        },
        "statistical_tests_run": num_stat_tests,
        "statistically_significant_changes": num_significant,
        "overall_assessment": (
            "Mitigation improved fairness" if improved_features
            else "Mitigation had mixed effects"
        ),
    }
