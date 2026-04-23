"""
AEGIS -- Model Evaluator
==========================

Computes both performance and fairness metrics for every trained model
produced by the Trainer module.

Performance metrics:
    - accuracy, precision, recall, F1

Fairness metrics:
    - demographic_parity_diff
    - equal_opportunity_diff
    - disparate_impact
    - fpr_gap   (false positive rate gap)
    - fnr_gap   (false negative rate gap)

Also supports:
    - Group-wise metric breakdown
    - Intersectional group metrics
    - Before vs after comparison tables
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .utils import get_logger, safe_divide

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

EvaluationResult = Dict[str, Any]
# {
#     "performance": {accuracy, precision, recall, f1},
#     "fairness": {demographic_parity_diff, equal_opportunity_diff, ...},
#     "group_metrics": {group_name: {accuracy, fpr, fnr, ...}},
#     "intersectional_metrics": {...},
# }


# ---------------------------------------------------------------------------
# Core Metric Functions
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard classification performance metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict with accuracy, precision, recall, f1.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-group classification metrics.

    For each group, computes:
        - accuracy
        - positive_rate  (predicted)
        - true_positive_rate  (TPR / recall)
        - false_positive_rate (FPR)
        - false_negative_rate (FNR)
        - count

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive: Group membership array.

    Returns:
        Nested dict: {group_name: {metric_name: value}}.
    """
    groups = np.unique(sensitive)
    result: Dict[str, Dict[str, float]] = {}

    for group in groups:
        mask = sensitive == group
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = float(((yt == 1) & (yp == 1)).sum())
        fp = float(((yt == 0) & (yp == 1)).sum())
        tn = float(((yt == 0) & (yp == 0)).sum())
        fn = float(((yt == 1) & (yp == 0)).sum())

        result[str(group)] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(yt, yp)) if len(yt) > 0 else 0.0,
            "positive_rate": safe_divide(float((yp == 1).sum()), float(len(yp))),
            "true_positive_rate": safe_divide(tp, tp + fn),
            "false_positive_rate": safe_divide(fp, fp + tn),
            "false_negative_rate": safe_divide(fn, fn + tp),
        }

    return result


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard fairness metrics.

    Metrics:
        - demographic_parity_diff: max - min positive rate across groups
        - equal_opportunity_diff: max - min TPR across groups
        - disparate_impact: min(positive_rate) / max(positive_rate)
        - fpr_gap: max - min FPR across groups
        - fnr_gap: max - min FNR across groups

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive: Group membership array.

    Returns:
        Dict of fairness metric values.
    """
    group_metrics = compute_group_metrics(y_true, y_pred, sensitive)

    positive_rates = [gm["positive_rate"] for gm in group_metrics.values()]
    tprs = [gm["true_positive_rate"] for gm in group_metrics.values()]
    fprs = [gm["false_positive_rate"] for gm in group_metrics.values()]
    fnrs = [gm["false_negative_rate"] for gm in group_metrics.values()]

    dp_diff = max(positive_rates) - min(positive_rates) if positive_rates else 0.0
    eo_diff = max(tprs) - min(tprs) if tprs else 0.0
    di_ratio = safe_divide(min(positive_rates), max(positive_rates), default=0.0) if positive_rates else 0.0
    fpr_gap = max(fprs) - min(fprs) if fprs else 0.0
    fnr_gap = max(fnrs) - min(fnrs) if fnrs else 0.0

    return {
        "demographic_parity_diff": round(dp_diff, 4),
        "equal_opportunity_diff": round(eo_diff, 4),
        "disparate_impact": round(di_ratio, 4),
        "fpr_gap": round(fpr_gap, 4),
        "fnr_gap": round(fnr_gap, 4),
    }


def compute_intersectional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for intersectional subgroups.

    Creates composite groups from all combinations of sensitive features
    and computes per-group metrics for each combination.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive_features: Dict mapping feature name -> array of group labels.

    Returns:
        Dict mapping composite group name -> metrics dict.
    """
    if len(sensitive_features) < 2:
        return {}

    # Build composite group labels
    feature_names = sorted(sensitive_features.keys())
    composite = pd.Series([""] * len(y_true))
    for name in feature_names:
        vals = sensitive_features[name]
        composite = composite + vals.astype(str) + "_"

    return compute_group_metrics(y_true, y_pred, composite.values)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    sensitive_features: Optional[Dict[str, np.ndarray]] = None,
) -> EvaluationResult:
    """
    Full evaluation of a single model.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive: Primary sensitive feature array.
        sensitive_features: Optional dict of all sensitive features
            (for intersectional analysis).

    Returns:
        EvaluationResult with performance, fairness, group, and
        intersectional metrics.
    """
    performance = compute_performance_metrics(y_true, y_pred)
    fairness = compute_fairness_metrics(y_true, y_pred, sensitive)
    group = compute_group_metrics(y_true, y_pred, sensitive)

    intersectional = {}
    if sensitive_features and len(sensitive_features) >= 2:
        intersectional = compute_intersectional_metrics(
            y_true, y_pred, sensitive_features
        )

    return {
        "performance": performance,
        "fairness": fairness,
        "group_metrics": group,
        "intersectional_metrics": intersectional,
    }


def evaluate_all_models(
    training_output: Dict[str, Any],
    sensitive_features: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, EvaluationResult]:
    """
    Evaluate every trained model from the Trainer output.

    Args:
        training_output: Dict mapping pipeline names to TrainedModel dicts.
        sensitive_features: Optional dict of all sensitive features.

    Returns:
        Dict mapping pipeline names to EvaluationResult dicts.
    """
    logger.info(f"Evaluating {len(training_output)} models ...")

    results: Dict[str, EvaluationResult] = {}

    for pipeline_name, trained in training_output.items():
        try:
            y_true = np.asarray(trained["y_test"])
            y_pred = np.asarray(trained["predictions"])
            sensitive = np.asarray(trained["sensitive_test"])

            # Build per-feature sensitive arrays for intersectional analysis
            feat_dict = None
            if sensitive_features:
                # Use test indices to slice
                test_idx = trained["sensitive_test"].index
                feat_dict = {}
                for feat_name, feat_arr in sensitive_features.items():
                    if isinstance(feat_arr, pd.Series):
                        try:
                            feat_dict[feat_name] = feat_arr.loc[test_idx].values
                        except KeyError:
                            feat_dict[feat_name] = np.asarray(feat_arr)[:len(y_true)]
                    else:
                        feat_dict[feat_name] = np.asarray(feat_arr)[:len(y_true)]

            evaluation = evaluate_model(y_true, y_pred, sensitive, feat_dict)
            results[pipeline_name] = evaluation

            logger.info(
                f"  [{pipeline_name}] "
                f"acc={evaluation['performance']['accuracy']:.4f}  "
                f"dp_diff={evaluation['fairness']['demographic_parity_diff']:.4f}  "
                f"di={evaluation['fairness']['disparate_impact']:.4f}"
            )

        except Exception as e:
            logger.error(f"Evaluation failed for '{pipeline_name}': {e}", exc_info=True)
            continue

    return results


def compare_before_after(
    baseline_eval: EvaluationResult,
    mitigated_eval: EvaluationResult,
) -> Dict[str, Any]:
    """
    Compute a before-vs-after comparison between baseline and mitigated
    evaluations.

    Args:
        baseline_eval: Evaluation result for the baseline (no mitigation).
        mitigated_eval: Evaluation result for the best mitigated model.

    Returns:
        Dict with deltas and improvement flags for each metric.
    """
    comparison: Dict[str, Any] = {}

    # Performance comparison
    for metric in ["accuracy", "precision", "recall", "f1"]:
        before = baseline_eval["performance"].get(metric, 0)
        after = mitigated_eval["performance"].get(metric, 0)
        comparison[f"{metric}_before"] = round(before, 4)
        comparison[f"{metric}_after"] = round(after, 4)
        comparison[f"{metric}_delta"] = round(after - before, 4)

    # Fairness comparison (lower is better for gap metrics)
    for metric in ["demographic_parity_diff", "equal_opportunity_diff", "fpr_gap", "fnr_gap"]:
        before = baseline_eval["fairness"].get(metric, 0)
        after = mitigated_eval["fairness"].get(metric, 0)
        comparison[f"{metric}_before"] = round(before, 4)
        comparison[f"{metric}_after"] = round(after, 4)
        comparison[f"{metric}_delta"] = round(after - before, 4)
        comparison[f"{metric}_improved"] = after < before

    # Disparate impact (higher is better, closer to 1.0)
    di_before = baseline_eval["fairness"].get("disparate_impact", 0)
    di_after = mitigated_eval["fairness"].get("disparate_impact", 0)
    comparison["disparate_impact_before"] = round(di_before, 4)
    comparison["disparate_impact_after"] = round(di_after, 4)
    comparison["disparate_impact_delta"] = round(di_after - di_before, 4)
    comparison["disparate_impact_improved"] = di_after > di_before

    # Overall fairness improvement score
    fairness_improvements = [
        v for k, v in comparison.items()
        if k.endswith("_improved") and isinstance(v, bool) and v
    ]
    comparison["fairness_improvement_count"] = len(fairness_improvements)

    return comparison
