"""
AEGIS -- Bias Classifier
=========================

Converts a raw bias report (from Phase 1 -- Bias Detection) into a structured
set of boolean bias tags.  Each tag indicates whether a specific type of bias
has been detected above a configurable threshold.

Supported bias categories:
    - representation_bias   -- group proportions are significantly imbalanced
    - outcome_bias          -- positive-outcome rates differ across groups
    - fairness_violation    -- standard fairness metrics are violated
    - proxy_bias            -- non-sensitive features are highly correlated with
                              sensitive attributes
    - intersectional_bias   -- bias at the intersection of multiple protected
                              attributes
    - label_bias            -- systematic noise or inconsistency in labels
                              across groups
"""

from typing import Any, Dict, List, Optional

from .utils import get_config, get_logger, safe_divide

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

BiasReport = Dict[str, Any]
BiasTags = Dict[str, bool]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_bias(
    bias_report: BiasReport,
    config: Optional[Dict[str, Any]] = None,
) -> BiasTags:
    """
    Analyse a raw bias report and emit boolean bias tags.

    Args:
        bias_report: Structured output from the Bias Detection phase.
            Expected keys (all optional -- missing sections are treated as
            "no bias detected for that category"):
                - distribution_bias
                - outcome_bias
                - fairness_metrics
                - advanced_bias
                - insights

        config: Optional overrides for detection thresholds.  Relevant keys:
                - representation_imbalance_threshold  (default 0.2)
                - outcome_disparity_threshold         (default 0.1)
                - fairness_violation_threshold        (default 0.05)
                - proxy_correlation_threshold         (default 0.3)
                - intersectional_threshold            (default 0.15)
                - label_noise_threshold               (default 0.1)

    Returns:
        Dictionary of boolean bias tags, e.g.::

            {
                "representation_bias": True,
                "outcome_bias": False,
                "fairness_violation": True,
                "proxy_bias": False,
                "intersectional_bias": False,
                "label_bias": False,
            }
    """
    cfg = get_config(config)
    logger.info("Classifying bias from report ...")

    tags: BiasTags = {
        "representation_bias": _check_representation_bias(
            bias_report, cfg["representation_imbalance_threshold"]
        ),
        "outcome_bias": _check_outcome_bias(
            bias_report, cfg["outcome_disparity_threshold"]
        ),
        "fairness_violation": _check_fairness_violation(
            bias_report, cfg["fairness_violation_threshold"]
        ),
        "proxy_bias": _check_proxy_bias(
            bias_report, cfg["proxy_correlation_threshold"]
        ),
        "intersectional_bias": _check_intersectional_bias(
            bias_report, cfg["intersectional_threshold"]
        ),
        "label_bias": _check_label_bias(
            bias_report, cfg["label_noise_threshold"]
        ),
    }

    detected = [k for k, v in tags.items() if v]
    logger.info(f"Bias tags detected: {detected if detected else 'none'}")
    return tags


def get_bias_summary(tags: BiasTags) -> str:
    """
    Return a human-readable summary of detected bias types.

    Args:
        tags: Output of `classify_bias`.

    Returns:
        Formatted summary string.
    """
    detected = [k.replace("_", " ").title() for k, v in tags.items() if v]
    if not detected:
        return "No significant bias detected."
    return "Detected bias types: " + ", ".join(detected) + "."


# ---------------------------------------------------------------------------
# Internal Classification Functions
# ---------------------------------------------------------------------------

def _check_representation_bias(
    report: BiasReport,
    threshold: float,
) -> bool:
    """
    Representation bias exists when the proportion gap between the largest
    and smallest group exceeds *threshold*.

    Examines ``report["distribution_bias"]``.
    """
    dist = report.get("distribution_bias", {})
    if not dist:
        return False

    # Iterate over each sensitive feature's distribution info
    for feature, info in dist.items():
        if isinstance(info, dict):
            # Expect keys like "group_proportions": {"male": 0.7, "female": 0.3}
            proportions = info.get("group_proportions", info.get("proportions", {}))
            if isinstance(proportions, dict) and len(proportions) >= 2:
                values = list(proportions.values())
                gap = max(values) - min(values)
                if gap > threshold:
                    logger.debug(
                        f"Representation bias in '{feature}': gap={gap:.3f} > {threshold}"
                    )
                    return True

            # Alternative: an explicit "imbalance_ratio" or "max_min_ratio"
            imbalance = info.get("imbalance_ratio", info.get("max_min_ratio", None))
            if imbalance is not None:
                # Normalise to 0-1 gap
                normalised = 1.0 - safe_divide(1.0, float(imbalance), default=1.0)
                if normalised > threshold:
                    return True

    return False


def _check_outcome_bias(
    report: BiasReport,
    threshold: float,
) -> bool:
    """
    Outcome bias exists when the positive-outcome rate differs across groups
    by more than *threshold*.

    Examines ``report["outcome_bias"]``.
    """
    outcome = report.get("outcome_bias", {})
    if not outcome:
        return False

    for feature, info in outcome.items():
        if isinstance(info, dict):
            rates = info.get("outcome_rates", info.get("positive_rates", {}))
            if isinstance(rates, dict) and len(rates) >= 2:
                values = list(rates.values())
                disparity = max(values) - min(values)
                if disparity > threshold:
                    logger.debug(
                        f"Outcome bias in '{feature}': disparity={disparity:.3f} > {threshold}"
                    )
                    return True

            # Direct "disparity" key
            disparity_val = info.get("disparity", info.get("outcome_disparity", None))
            if disparity_val is not None and abs(float(disparity_val)) > threshold:
                return True

    return False


def _check_fairness_violation(
    report: BiasReport,
    threshold: float,
) -> bool:
    """
    A fairness violation is flagged when standard fairness metrics
    (demographic parity difference, equal opportunity difference, etc.)
    exceed *threshold*.

    Examines ``report["fairness_metrics"]``.
    """
    fairness = report.get("fairness_metrics", {})
    if not fairness:
        return False

    # May be nested per-feature or flat
    metrics_to_check = [
        "demographic_parity_difference",
        "demographic_parity_diff",
        "equal_opportunity_difference",
        "equal_opportunity_diff",
        "equalized_odds_difference",
        "equalized_odds_diff",
        "disparate_impact_ratio",
    ]

    def _scan(d: Dict[str, Any]) -> bool:
        """Recursively scan dict for any violating metric."""
        for key, val in d.items():
            if isinstance(val, dict):
                if _scan(val):
                    return True
            elif key.lower() in metrics_to_check:
                try:
                    numeric = abs(float(val))
                    # For disparate impact ratio, violation is when < 0.8
                    if "disparate_impact" in key.lower():
                        if numeric < (1.0 - threshold) or numeric < 0.8:
                            return True
                    elif numeric > threshold:
                        return True
                except (ValueError, TypeError):
                    continue
        return False

    return _scan(fairness)


def _check_proxy_bias(
    report: BiasReport,
    threshold: float,
) -> bool:
    """
    Proxy bias exists when non-sensitive features are highly correlated
    with sensitive attributes (correlation > *threshold*).

    Examines ``report["advanced_bias"]["proxy_bias"]`` or
    ``report["advanced_bias"]["proxy_features"]``.
    """
    advanced = report.get("advanced_bias", {})
    if not advanced:
        return False

    proxy_info = advanced.get("proxy_bias", advanced.get("proxy_features", {}))
    if not proxy_info:
        return False

    if isinstance(proxy_info, dict):
        for feature, details in proxy_info.items():
            if isinstance(details, dict):
                corr = details.get("correlation", details.get("mutual_information", 0))
                if abs(float(corr)) > threshold:
                    logger.debug(f"Proxy bias detected via '{feature}': corr={corr}")
                    return True
            elif isinstance(details, (int, float)):
                if abs(float(details)) > threshold:
                    return True

    elif isinstance(proxy_info, list):
        # List of proxy feature dicts
        for item in proxy_info:
            if isinstance(item, dict):
                corr = item.get("correlation", item.get("score", 0))
                if abs(float(corr)) > threshold:
                    return True

    return False


def _check_intersectional_bias(
    report: BiasReport,
    threshold: float,
) -> bool:
    """
    Intersectional bias is present when bias at the intersection of
    multiple protected attributes is worse than for any single attribute.

    Examines ``report["advanced_bias"]["intersectional_bias"]``.
    """
    advanced = report.get("advanced_bias", {})
    if not advanced:
        return False

    inter = advanced.get("intersectional_bias", {})
    if not inter:
        return False

    if isinstance(inter, dict):
        # Look for disparity values
        for key, val in inter.items():
            if isinstance(val, dict):
                disparity = val.get("disparity", val.get("gap", 0))
                if abs(float(disparity)) > threshold:
                    return True
            elif isinstance(val, (int, float)):
                if abs(float(val)) > threshold:
                    return True

    return False


def _check_label_bias(
    report: BiasReport,
    threshold: float,
) -> bool:
    """
    Label bias exists when label noise or inconsistency differs
    significantly across groups.

    Examines ``report["advanced_bias"]["label_bias"]``.
    """
    advanced = report.get("advanced_bias", {})
    if not advanced:
        return False

    label_info = advanced.get("label_bias", {})
    if not label_info:
        return False

    if isinstance(label_info, dict):
        for key, val in label_info.items():
            if isinstance(val, dict):
                noise = val.get("noise_rate", val.get("inconsistency", 0))
                if abs(float(noise)) > threshold:
                    return True
            elif isinstance(val, (int, float)):
                if abs(float(val)) > threshold:
                    return True

    return False
