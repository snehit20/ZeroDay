"""
AEGIS -- Strategy Selector
============================

Maps boolean bias tags (from the Classifier) to relevant mitigation
strategies.  Each bias type has a primary strategy, one or more secondary
strategies, and an associated human-readable reason explaining *why* that
strategy is appropriate.

The selector also supports *combination strategies* -- when multiple bias
types are detected simultaneously, it can suggest composite pipelines.
"""

from typing import Any, Dict, List, Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

BiasTags = Dict[str, bool]

StrategyEntry = Dict[str, Any]
# {
#     "primary": str,
#     "secondary": List[str],
#     "reason": str,
# }

SelectionResult = Dict[str, Any]
# {
#     "per_bias_strategies": Dict[str, StrategyEntry],
#     "combined_strategies": List[str],
#     "all_unique_strategies": List[str],
#     "reasons": Dict[str, str],
# }


# ---------------------------------------------------------------------------
# Strategy Mapping Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, StrategyEntry] = {
    "representation_bias": {
        "primary": "reweighting",
        "secondary": ["resampling", "smote"],
        "reason": (
            "Representation bias indicates significant group imbalance in the "
            "training data.  Reweighting assigns higher importance to under-"
            "represented groups, while resampling/SMOTE directly corrects "
            "class distribution."
        ),
    },
    "outcome_bias": {
        "primary": "threshold_optimization",
        "secondary": ["reweighting", "fairlearn_reduction"],
        "reason": (
            "Outcome bias means positive-outcome rates differ across groups.  "
            "Group-specific threshold optimization directly equalises acceptance "
            "rates, while reweighting and constrained optimisation offer "
            "complementary corrections."
        ),
    },
    "fairness_violation": {
        "primary": "fairlearn_reduction",
        "secondary": ["threshold_optimization", "reweighting"],
        "reason": (
            "Fairness violations indicate that standard metrics (demographic "
            "parity, equal opportunity) are breached.  Fairlearn's constrained "
            "reduction approach directly optimises for these constraints."
        ),
    },
    "proxy_bias": {
        "primary": "disparate_impact_remover",
        "secondary": ["reweighting"],
        "reason": (
            "Proxy bias arises when non-sensitive features act as proxies for "
            "protected attributes.  Disparate Impact Remover modifies feature "
            "distributions to remove this dependence."
        ),
    },
    "intersectional_bias": {
        "primary": "reweighting",
        "secondary": ["fairlearn_reduction", "resampling"],
        "reason": (
            "Intersectional bias is harder to address with single-axis "
            "corrections.  Reweighting at the intersectional subgroup level, "
            "combined with constrained optimisation, provides the best coverage."
        ),
    },
    "label_bias": {
        "primary": "reweighting",
        "secondary": ["resampling"],
        "reason": (
            "Label bias suggests systematic label noise that differs across "
            "groups.  Reweighting down-weights potentially noisy subgroups, "
            "while resampling can create a more balanced label distribution."
        ),
    },
}

# Pre-defined beneficial combinations
COMBINATION_RULES: List[Tuple[List[str], str]] = [
    # (required bias types, combined strategy name)
    (
        ["representation_bias", "outcome_bias"],
        "reweighting + threshold_optimization",
    ),
    (
        ["proxy_bias", "representation_bias"],
        "disparate_impact_remover + reweighting",
    ),
    (
        ["proxy_bias", "fairness_violation"],
        "disparate_impact_remover + fairlearn_reduction",
    ),
    (
        ["representation_bias", "fairness_violation"],
        "reweighting + fairlearn_reduction",
    ),
    (
        ["outcome_bias", "fairness_violation"],
        "threshold_optimization + fairlearn_reduction",
    ),
    (
        ["proxy_bias", "outcome_bias"],
        "disparate_impact_remover + threshold_optimization",
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_strategies(
    bias_tags: BiasTags,
    custom_registry: Optional[Dict[str, StrategyEntry]] = None,
) -> SelectionResult:
    """
    Map bias tags to mitigation strategies.

    Args:
        bias_tags: Dictionary of boolean bias tags from the classifier.
        custom_registry: Optional override for the default strategy registry.

    Returns:
        SelectionResult dict containing:
            - per_bias_strategies: strategies mapped to each detected bias
            - combined_strategies: composite strategies for co-occurring biases
            - all_unique_strategies: deduplicated flat list of all strategies
            - reasons: human-readable reason per strategy
    """
    registry = custom_registry or STRATEGY_REGISTRY
    detected = [bias for bias, present in bias_tags.items() if present]

    logger.info(f"Selecting strategies for detected biases: {detected}")

    # --- Per-bias strategies ---
    per_bias: Dict[str, StrategyEntry] = {}
    all_strategies: List[str] = []
    reasons: Dict[str, str] = {}

    for bias_type in detected:
        entry = registry.get(bias_type)
        if entry is None:
            logger.warning(f"No strategy registered for bias type: {bias_type}")
            continue

        per_bias[bias_type] = entry
        all_strategies.append(entry["primary"])
        all_strategies.extend(entry["secondary"])
        reasons[entry["primary"]] = entry["reason"]

    # --- Combination strategies ---
    combined: List[str] = []
    for required_biases, combo_name in COMBINATION_RULES:
        if all(b in detected for b in required_biases):
            combined.append(combo_name)
            reasons[combo_name] = (
                f"Combined strategy targeting co-occurring biases: "
                f"{', '.join(b.replace('_', ' ') for b in required_biases)}."
            )

    all_strategies.extend(combined)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for s in all_strategies:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    result: SelectionResult = {
        "per_bias_strategies": per_bias,
        "combined_strategies": combined,
        "all_unique_strategies": unique,
        "reasons": reasons,
    }

    logger.info(f"Selected {len(unique)} unique strategies (incl. {len(combined)} combinations)")
    return result


def get_strategy_reason(
    strategy_name: str,
    selection_result: Optional[SelectionResult] = None,
) -> str:
    """
    Return the human-readable reason for a given strategy.

    Args:
        strategy_name: Name of the strategy.
        selection_result: Optional selection result to look up reasons.

    Returns:
        Reason string, or a default message if not found.
    """
    if selection_result and strategy_name in selection_result.get("reasons", {}):
        return selection_result["reasons"][strategy_name]

    # Fall back to registry
    for entry in STRATEGY_REGISTRY.values():
        if entry["primary"] == strategy_name:
            return entry["reason"]

    return f"Strategy '{strategy_name}' selected based on detected bias patterns."
