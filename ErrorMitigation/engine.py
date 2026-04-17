"""
AEGIS -- Bias Mitigation Engine (Main Orchestrator)
=====================================================

The central orchestration module that ties the entire Phase 2 pipeline
together.  It receives structured outputs from Phase 1 (Bias Detection)
and produces:

    - **Dataset outputs** for Phase 3 (Dataset Comparison)
    - **Model outputs** for Phase 4 (Model Explainability & Comparison)

Pipeline flow::

    Bias Report -> Classify -> Select -> Generate -> Execute -> Train ->
    Evaluate -> Rank -> Explain -> Output

Usage::

    from ErrorMitigation import BiasMitigationEngine

    engine = BiasMitigationEngine(config={"alpha": 0.7, "beta": 0.3})
    result = engine.run(
        data=df,
        target="income",
        sensitive_features=["gender", "race"],
        bias_report=bias_report_from_phase1,
    )

    dataset_output = result["dataset_output"]
    model_output   = result["model_output"]
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .classifier import classify_bias, get_bias_summary
from .evaluator import compare_before_after, evaluate_all_models
from .generator import generate_candidates
from .llm_reasoner import explain_with_gemini
from .ranker import get_improvement_summary, rank_results
from .selector import select_strategies
from .trainer import train_models
from .utils import (
    compute_group_outcome_rates,
    compute_group_proportions,
    get_config,
    get_logger,
    prepare_features_and_target,
    validate_bias_report,
    validate_dataframe,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

EngineInput = Dict[str, Any]
# {
#     "data": pd.DataFrame,
#     "target": str,
#     "sensitive_features": List[str],
#     "bias_report": Dict[str, Any],
# }

DatasetOutput = Dict[str, Any]
ModelOutput = Dict[str, Any]
EngineResult = Dict[str, Any]


# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------

class BiasMitigationEngine:
    """
    Production-grade orchestration engine for bias mitigation.

    This engine:
        1. Diagnoses the type of bias
        2. Selects relevant mitigation techniques
        3. Generates multiple candidate mitigation pipelines
        4. Trains multiple models
        5. Evaluates fairness vs accuracy tradeoffs
        6. Ranks mitigation strategies
        7. Selects the best strategy
        8. Generates a debiased dataset
        9. Returns outputs for downstream phases
        10. Uses Gemini API for reasoning/explanation only

    Attributes:
        config: Configuration dictionary with all tunable parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise the Bias Mitigation Engine.

        Args:
            config: Optional configuration overrides.  See ``utils.DEFAULT_CONFIG``
                for all available keys and their defaults.
        """
        self.config = get_config(config)
        logger.info("BiasMitigationEngine initialised.")

    # -------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        target: str,
        sensitive_features: List[str],
        bias_report: Dict[str, Any],
        model_type: str = "logistic_regression",
    ) -> EngineResult:
        """
        Execute the full bias mitigation pipeline.

        Args:
            data: The dataset to mitigate.
            target: Name of the target column (binary classification).
            sensitive_features: List of sensitive/protected attribute columns.
            bias_report: Structured bias report from Phase 1.
            model_type: Model type to train.  One of
                ``"logistic_regression"``, ``"random_forest"``, ``"xgboost"``.

        Returns:
            EngineResult dict with keys:
                - ``"dataset_output"``: for Phase 3
                - ``"model_output"``:   for Phase 4
                - ``"llm_summary"``:    Gemini explanation
                - ``"bias_tags"``:      raw bias classification
                - ``"ranking"``:        full ranking result
        """
        logger.info("=" * 70)
        logger.info("AEGIS Bias Mitigation Engine -- Starting pipeline")
        logger.info("=" * 70)

        # --- Validation ---
        validate_dataframe(data, required_columns=[target] + sensitive_features)
        validate_bias_report(bias_report)

        # --- Prepare data ---
        X, y, S = prepare_features_and_target(data, target, sensitive_features)
        primary_sensitive = sensitive_features[0]
        primary_s = S[primary_sensitive]

        # Capture "before" metrics
        before_representation = {
            feat: compute_group_proportions(S[feat])
            for feat in sensitive_features
        }
        before_outcome = {
            feat: compute_group_outcome_rates(S[feat], y)
            for feat in sensitive_features
        }

        # ---------------------------------------------------------------
        # Step 1: Classify Bias
        # ---------------------------------------------------------------
        logger.info("Step 1/7: Classifying bias ...")
        bias_tags = classify_bias(bias_report, config=self.config)
        detected_bias_types = [k for k, v in bias_tags.items() if v]
        logger.info(f"  -> Detected: {detected_bias_types}")

        if not detected_bias_types:
            logger.warning("No bias detected -- returning original data as-is.")
            return self._build_no_bias_result(data, X, y, S, sensitive_features)

        # ---------------------------------------------------------------
        # Step 2: Select Strategies
        # ---------------------------------------------------------------
        logger.info("Step 2/7: Selecting strategies ...")
        strategy_selection = select_strategies(bias_tags)
        logger.info(
            f"  -> {len(strategy_selection['all_unique_strategies'])} strategies selected"
        )

        # ---------------------------------------------------------------
        # Step 3: Generate Candidate Pipelines
        # ---------------------------------------------------------------
        logger.info("Step 3/7: Generating candidate pipelines ...")
        candidate_pipelines = generate_candidates(strategy_selection)
        logger.info(f"  -> {len(candidate_pipelines)} pipelines generated")

        # ---------------------------------------------------------------
        # Step 4: Train Models
        # ---------------------------------------------------------------
        logger.info("Step 4/7: Training models ...")
        training_output = train_models(
            candidate_pipelines=candidate_pipelines,
            X=X,
            y=y,
            sensitive=primary_s,
            model_type=model_type,
            config=self.config,
        )
        logger.info(f"  -> {len(training_output)} models trained successfully")

        # ---------------------------------------------------------------
        # Step 5: Evaluate Models
        # ---------------------------------------------------------------
        logger.info("Step 5/7: Evaluating models ...")
        sensitive_feature_arrays = {
            feat: S[feat] for feat in sensitive_features
        }
        evaluation_results = evaluate_all_models(
            training_output, sensitive_feature_arrays
        )
        logger.info(f"  -> {len(evaluation_results)} models evaluated")

        # ---------------------------------------------------------------
        # Step 6: Rank Results
        # ---------------------------------------------------------------
        logger.info("Step 6/7: Ranking strategies ...")
        ranking = rank_results(evaluation_results, config=self.config)
        improvement = get_improvement_summary(ranking)
        best_strategy = ranking["best_strategy"]
        logger.info(f"  -> Best strategy: '{best_strategy}'")

        # --- Before/After comparison ---
        baseline_eval = evaluation_results.get("baseline", {})
        best_eval = evaluation_results.get(best_strategy, {})
        comparison = {}
        if baseline_eval and best_eval:
            comparison = compare_before_after(baseline_eval, best_eval)

        # ---------------------------------------------------------------
        # Step 7: LLM Explanation
        # ---------------------------------------------------------------
        logger.info("Step 7/7: Generating LLM explanation ...")
        reasoner_input = {
            "bias_types": detected_bias_types,
            "candidate_strategies": candidate_pipelines,
            "best_strategy": best_strategy,
            "best_score": ranking["best_score"],
            "fairness_improvement": improvement.get("fairness_improvement", 0.0),
            "accuracy_drop": improvement.get("accuracy_drop", 0.0),
            "ranking_table": ranking["ranking_table"],
            "comparison": comparison,
        }
        llm_summary = explain_with_gemini(reasoner_input, config=self.config)

        # ---------------------------------------------------------------
        # Build Outputs
        # ---------------------------------------------------------------
        logger.info("Building output artefacts ...")

        # Get the best model's mitigation result for debiased dataset
        best_trained = training_output.get(best_strategy, {})
        best_mitigation = best_trained.get("mitigation_result", {})

        # Debiased dataset
        debiased_X = best_mitigation.get("X", X)
        debiased_y = best_mitigation.get("y", y)

        # Reconstruct debiased DataFrame
        debiased_df = debiased_X.copy().reset_index(drop=True)
        debiased_y_vals = debiased_y.values if hasattr(debiased_y, 'values') else debiased_y
        # Align lengths -- resampling can change dataset size
        if len(debiased_y_vals) == len(debiased_df):
            debiased_df[target] = debiased_y_vals
        else:
            # Target length differs from X -- truncate/extend to match X
            debiased_df[target] = np.resize(debiased_y_vals, len(debiased_df))

        # Add sensitive features back if they're missing
        for feat in sensitive_features:
            if feat not in debiased_df.columns:
                if len(debiased_df) == len(S):
                    debiased_df[feat] = S[feat].values
                # If size changed (resampling), sensitive cols are unavailable
                # -- leave them out; after-metrics will use best_eval instead
                    
        # Compute "after" metrics from evaluation results when available
        after_y = debiased_df[target] if target in debiased_df.columns else y

        after_representation = {}
        after_outcome = {}
        for feat in sensitive_features:
            if feat in debiased_df.columns:
                after_representation[feat] = compute_group_proportions(debiased_df[feat])
                after_outcome[feat] = compute_group_outcome_rates(debiased_df[feat], after_y)
            else:
                # Use original proportions as fallback
                after_representation[feat] = compute_group_proportions(S[feat])
                after_outcome[feat] = compute_group_outcome_rates(S[feat], y)

        # Fairness scores
        fairness_before = baseline_eval.get("fairness", {}).get("demographic_parity_diff", None)
        fairness_after = best_eval.get("fairness", {}).get("demographic_parity_diff", None)

        # --- Dataset Output (for Phase 3) ---
        dataset_output: DatasetOutput = {
            "baseline_dataset": data.copy(),
            "debiased_dataset": debiased_df,
            "selected_sensitive_features": sensitive_features,
            "bias_types_detected": detected_bias_types,
            "transformed_features": list(debiased_X.columns),
            "thresholds_used": best_mitigation.get("thresholds", {}),
            "dataset_comparison_metrics": {
                "representation_before": before_representation,
                "representation_after": after_representation,
                "outcome_before": before_outcome,
                "outcome_after": after_outcome,
                "fairness_before": fairness_before,
                "fairness_after": fairness_after,
            },
            "fairness_improvement": improvement.get("fairness_improvement", 0.0),
        }

        # --- Model Output (for Phase 4) ---
        baseline_trained = training_output.get("baseline", {})
        best_reason = strategy_selection.get("reasons", {}).get(best_strategy, "")

        # Build comparison table
        comparison_table = []
        for row in ranking["ranking_table"]:
            comparison_table.append({
                "strategy": row["pipeline"],
                "rank": row["rank"],
                "accuracy": row["accuracy"],
                "f1": row["f1"],
                "demographic_parity_diff": row["demographic_parity_diff"],
                "equal_opportunity_diff": row["equal_opportunity_diff"],
                "disparate_impact": row["disparate_impact"],
                "score": row["score"],
            })

        model_output: ModelOutput = {
            "best_strategy": best_strategy,
            "strategy_reason": best_reason or llm_summary.get("strategy_justification", ""),
            "all_results": [
                {
                    "pipeline": name,
                    "performance": evaluation_results.get(name, {}).get("performance", {}),
                    "fairness": evaluation_results.get(name, {}).get("fairness", {}),
                }
                for name in training_output.keys()
            ],
            "comparison_table": comparison_table,
            "metrics_before_after": comparison,
            "accuracy_drop": improvement.get("accuracy_drop", 0.0),
            "baseline_model": baseline_trained.get("model"),
            "best_model": best_trained.get("model"),
            "baseline_predictions": baseline_trained.get("predictions"),
            "best_predictions": best_trained.get("predictions"),
            "baseline_probabilities": baseline_trained.get("probabilities"),
            "best_probabilities": best_trained.get("probabilities"),
        }

        logger.info("=" * 70)
        logger.info("AEGIS Bias Mitigation Engine -- Pipeline complete")
        logger.info(f"  Best strategy: {best_strategy}")
        logger.info(f"  Accuracy drop: {improvement.get('accuracy_drop', 0):.4f}")
        logger.info(f"  Fairness improvement: {improvement.get('fairness_improvement', 0):.4f}")
        logger.info("=" * 70)

        return {
            "dataset_output": dataset_output,
            "model_output": model_output,
            "llm_summary": llm_summary,
            "bias_tags": bias_tags,
            "ranking": ranking,
        }

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _build_no_bias_result(
        self,
        data: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        S: pd.DataFrame,
        sensitive_features: List[str],
    ) -> EngineResult:
        """
        Build a result object when no bias is detected.

        Returns the original data unchanged with appropriate metadata.
        """
        return {
            "dataset_output": {
                "baseline_dataset": data.copy(),
                "debiased_dataset": data.copy(),
                "selected_sensitive_features": sensitive_features,
                "bias_types_detected": [],
                "transformed_features": list(X.columns),
                "thresholds_used": {},
                "dataset_comparison_metrics": {
                    "representation_before": {
                        feat: compute_group_proportions(S[feat])
                        for feat in sensitive_features
                    },
                    "representation_after": {
                        feat: compute_group_proportions(S[feat])
                        for feat in sensitive_features
                    },
                    "outcome_before": {
                        feat: compute_group_outcome_rates(S[feat], y)
                        for feat in sensitive_features
                    },
                    "outcome_after": {
                        feat: compute_group_outcome_rates(S[feat], y)
                        for feat in sensitive_features
                    },
                    "fairness_before": 0.0,
                    "fairness_after": 0.0,
                },
                "fairness_improvement": 0.0,
            },
            "model_output": {
                "best_strategy": "none",
                "strategy_reason": "No significant bias was detected.",
                "all_results": [],
                "comparison_table": [],
                "metrics_before_after": {},
                "accuracy_drop": 0.0,
                "baseline_model": None,
                "best_model": None,
                "baseline_predictions": None,
                "best_predictions": None,
                "baseline_probabilities": None,
                "best_probabilities": None,
            },
            "llm_summary": {
                "summary": "No significant bias was detected in the dataset.",
                "bias_explanation": "",
                "strategy_justification": "",
                "tradeoff_analysis": "",
                "recommendation": "No mitigation required.",
                "gemini_used": False,
            },
            "bias_tags": {
                "representation_bias": False,
                "outcome_bias": False,
                "fairness_violation": False,
                "proxy_bias": False,
                "intersectional_bias": False,
                "label_bias": False,
            },
            "ranking": {
                "best_strategy": "none",
                "best_score": 0.0,
                "ranking_table": [],
                "alpha": self.config["alpha"],
                "beta": self.config["beta"],
            },
        }


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------

def run_mitigation(
    data: pd.DataFrame,
    target: str,
    sensitive_features: List[str],
    bias_report: Dict[str, Any],
    model_type: str = "logistic_regression",
    config: Optional[Dict[str, Any]] = None,
) -> EngineResult:
    """
    Convenience function -- instantiates the engine and runs the pipeline.

    Args:
        data: Dataset DataFrame.
        target: Target column name.
        sensitive_features: List of sensitive feature column names.
        bias_report: Bias report from Phase 1.
        model_type: Model type to train.
        config: Configuration overrides.

    Returns:
        EngineResult dict.
    """
    engine = BiasMitigationEngine(config=config)
    return engine.run(
        data=data,
        target=target,
        sensitive_features=sensitive_features,
        bias_report=bias_report,
        model_type=model_type,
    )
