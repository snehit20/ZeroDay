"""
AEGIS -- Strategy Ranker
=========================

Ranks all candidate mitigation strategies by a configurable
fairness-accuracy tradeoff score:

    score = alpha * accuracy - beta * unfairness

where *unfairness* is the demographic parity difference (or any other
chosen fairness gap metric).

Returns the best strategy, its score, and a full ranking table suitable
for display, logging, or downstream consumption.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import get_config, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

RankingResult = Dict[str, Any]
# {
#     "best_strategy": str,
#     "best_score": float,
#     "ranking_table": List[Dict[str, Any]],
#     "alpha": float,
#     "beta": float,
# }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_results(
    evaluation_results: Dict[str, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    fairness_metric: str = "demographic_parity_diff",
    performance_metric: str = "accuracy",
) -> RankingResult:
    """
    Rank all evaluated pipelines by the fairness-accuracy tradeoff score.

    Score formula::

        score = alpha * performance - beta * unfairness

    Where:
        - ``performance`` = value of ``performance_metric`` (higher is better)
        - ``unfairness``  = value of ``fairness_metric``    (lower is better)

    Args:
        evaluation_results: Dict mapping pipeline names to evaluation
            result dicts (output of ``evaluator.evaluate_all_models``).
        config: Configuration overrides.  Relevant keys:
            - ``alpha`` (default 0.6): weight for performance
            - ``beta``  (default 0.4): weight for unfairness penalty
        fairness_metric: Which fairness metric to use as the unfairness
            penalty.  Default: ``"demographic_parity_diff"``.
        performance_metric: Which performance metric to use.
            Default: ``"accuracy"``.

    Returns:
        RankingResult dict with best strategy, best score, and full
        ranking table sorted descending by score.
    """
    cfg = get_config(config)
    alpha = cfg.get("alpha", 0.6)
    beta = cfg.get("beta", 0.4)

    logger.info(
        f"Ranking {len(evaluation_results)} strategies "
        f"(a={alpha}, b={beta}, perf={performance_metric}, fair={fairness_metric})"
    )

    rows: List[Dict[str, Any]] = []

    for pipeline, evaluation in evaluation_results.items():
        perf = evaluation.get("performance", {})
        fair = evaluation.get("fairness", {})

        accuracy = perf.get(performance_metric, 0.0)
        unfairness = fair.get(fairness_metric, 0.0)

        score = alpha * accuracy - beta * unfairness

        row = {
            "pipeline": pipeline,
            "accuracy": round(accuracy, 4),
            "precision": round(perf.get("precision", 0.0), 4),
            "recall": round(perf.get("recall", 0.0), 4),
            "f1": round(perf.get("f1", 0.0), 4),
            "demographic_parity_diff": round(fair.get("demographic_parity_diff", 0.0), 4),
            "equal_opportunity_diff": round(fair.get("equal_opportunity_diff", 0.0), 4),
            "disparate_impact": round(fair.get("disparate_impact", 0.0), 4),
            "fpr_gap": round(fair.get("fpr_gap", 0.0), 4),
            "fnr_gap": round(fair.get("fnr_gap", 0.0), 4),
            "unfairness": round(unfairness, 4),
            "score": round(score, 4),
        }
        rows.append(row)

    # Sort by score descending
    rows.sort(key=lambda r: r["score"], reverse=True)

    # Assign ranks
    for i, row in enumerate(rows):
        row["rank"] = i + 1

    best = rows[0] if rows else {"pipeline": "none", "score": 0.0}

    logger.info(
        f"Best strategy: '{best['pipeline']}' "
        f"(score={best['score']:.4f}, acc={best.get('accuracy', 0):.4f}, "
        f"unfairness={best.get('unfairness', 0):.4f})"
    )

    # Log full table
    if logger.isEnabledFor(20):  # INFO
        for row in rows:
            logger.info(
                f"  #{row['rank']} {row['pipeline']:40s} "
                f"score={row['score']:.4f}  acc={row['accuracy']:.4f}  "
                f"dp_diff={row['demographic_parity_diff']:.4f}"
            )

    return {
        "best_strategy": best["pipeline"],
        "best_score": best["score"],
        "ranking_table": rows,
        "alpha": alpha,
        "beta": beta,
    }


def get_ranking_dataframe(ranking_result: RankingResult) -> pd.DataFrame:
    """
    Convert the ranking table to a pandas DataFrame for easy display.

    Args:
        ranking_result: Output of ``rank_results``.

    Returns:
        DataFrame with one row per pipeline, sorted by rank.
    """
    return pd.DataFrame(ranking_result["ranking_table"]).set_index("rank")


def get_improvement_summary(
    ranking_result: RankingResult,
) -> Dict[str, Any]:
    """
    Compute a summary comparing the best strategy to baseline.

    Args:
        ranking_result: Output of ``rank_results``.

    Returns:
        Dict with accuracy_drop, fairness_improvement, and other deltas.
    """
    table = ranking_result["ranking_table"]

    baseline = next((r for r in table if r["pipeline"] == "baseline"), None)
    best = table[0] if table else None

    if baseline is None or best is None:
        return {
            "accuracy_drop": 0.0,
            "fairness_improvement": 0.0,
            "has_baseline": baseline is not None,
        }

    accuracy_drop = baseline["accuracy"] - best["accuracy"]
    fairness_improvement = baseline["unfairness"] - best["unfairness"]

    return {
        "best_strategy": best["pipeline"],
        "baseline_accuracy": baseline["accuracy"],
        "best_accuracy": best["accuracy"],
        "accuracy_drop": round(accuracy_drop, 4),
        "baseline_unfairness": baseline["unfairness"],
        "best_unfairness": best["unfairness"],
        "fairness_improvement": round(fairness_improvement, 4),
        "score_improvement": round(best["score"] - baseline["score"], 4),
    }
