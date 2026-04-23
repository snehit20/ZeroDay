"""
AEGIS -- Candidate Pipeline Generator
=======================================

Takes the selection result from the Selector and produces an ordered list
of candidate mitigation pipelines to be executed by the Trainer.

Pipelines can be:
    - Single strategies  (e.g. ``"reweighting"``)
    - Combined strategies (e.g. ``"disparate_impact_remover + reweighting"``)

The generator ensures:
    1. A no-mitigation **baseline** is always included first.
    2. Primary strategies come before secondary ones.
    3. Combinations are placed after their constituent singles.
    4. Duplicates are removed.
"""

from typing import Any, Dict, List, Optional

from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

SelectionResult = Dict[str, Any]
CandidatePipeline = str  # e.g. "reweighting" or "dir + reweighting"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_candidates(
    selection_result: SelectionResult,
    max_candidates: int = 10,
    include_baseline: bool = True,
) -> List[CandidatePipeline]:
    """
    Generate an ordered list of candidate mitigation pipelines.

    Args:
        selection_result: Output from ``selector.select_strategies()``.
            Expected keys:
                - per_bias_strategies
                - combined_strategies
                - all_unique_strategies

        max_candidates: Maximum number of pipelines to return (including
            baseline).  Set to -1 for unlimited.

        include_baseline: Whether to include a ``"baseline"`` (no mitigation)
            pipeline as the first entry.

    Returns:
        Ordered list of pipeline name strings.
    """
    logger.info("Generating candidate mitigation pipelines ...")

    candidates: List[CandidatePipeline] = []

    # --- 1. Baseline (no mitigation) ---
    if include_baseline:
        candidates.append("baseline")

    # --- 2. Primary strategies (one per detected bias type) ---
    per_bias = selection_result.get("per_bias_strategies", {})
    for bias_type, entry in per_bias.items():
        primary = entry.get("primary")
        if primary and primary not in candidates:
            candidates.append(primary)

    # --- 3. Secondary strategies ---
    for bias_type, entry in per_bias.items():
        for secondary in entry.get("secondary", []):
            if secondary not in candidates:
                candidates.append(secondary)

    # --- 4. Combination strategies ---
    for combo in selection_result.get("combined_strategies", []):
        if combo not in candidates:
            candidates.append(combo)

    # --- 5. Any remaining unique strategies not yet included ---
    for strategy in selection_result.get("all_unique_strategies", []):
        if strategy not in candidates:
            candidates.append(strategy)

    # --- Enforce cap ---
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    logger.info(
        f"Generated {len(candidates)} candidate pipelines: {candidates}"
    )
    return candidates


def parse_pipeline_steps(pipeline: CandidatePipeline) -> List[str]:
    """
    Parse a pipeline string into individual strategy steps.

    Examples::

        "reweighting"                          -> ["reweighting"]
        "disparate_impact_remover + reweighting" -> ["disparate_impact_remover", "reweighting"]
        "dir + reweighting"                    -> ["disparate_impact_remover", "reweighting"]

    Args:
        pipeline: Pipeline name string.

    Returns:
        List of individual strategy names.
    """
    # Normalise common abbreviations
    ALIASES = {
        "dir": "disparate_impact_remover",
        "di_remover": "disparate_impact_remover",
        "smote": "smote",
        "fl_reduction": "fairlearn_reduction",
        "threshold_opt": "threshold_optimization",
    }

    steps = [s.strip() for s in pipeline.split("+")]
    normalised: List[str] = []
    for step in steps:
        normalised.append(ALIASES.get(step, step))
    return normalised


def describe_pipeline(pipeline: CandidatePipeline) -> str:
    """
    Return a human-readable description of a pipeline.

    Args:
        pipeline: Pipeline name string.

    Returns:
        Description string.
    """
    DESCRIPTIONS = {
        "baseline": "No mitigation applied -- serves as the control model.",
        "reweighting": "Assign sample weights inversely proportional to group prevalence.",
        "resampling": "Over/under-sample minority/majority groups to balance representation.",
        "smote": "Generate synthetic minority samples via SMOTE interpolation.",
        "disparate_impact_remover": "Transform features to remove dependence on sensitive attributes.",
        "threshold_optimization": "Optimise group-specific decision thresholds to equalise outcomes.",
        "fairlearn_reduction": "Train under fairness constraints via Exponentiated Gradient reduction.",
    }

    steps = parse_pipeline_steps(pipeline)
    descriptions = []
    for step in steps:
        desc = DESCRIPTIONS.get(step, f"Apply {step.replace('_', ' ')} technique.")
        descriptions.append(desc)

    return " -> ".join(descriptions)
