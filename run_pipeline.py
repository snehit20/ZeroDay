"""
AEGIS -- Automated Pipeline Runner
=====================================

Fully automated end-to-end bias detection and mitigation pipeline.

Phases:
    Phase 1: Bias Detection  (built-in analyzer)
    Phase 2: Bias Mitigation (ErrorMitigation engine)
    Phase 3: Artifact Export  (CSV, JSON, model pickle)

Usage:
    # Run with synthetic data (demo mode):
    python run_pipeline.py

    # Run with your own CSV:
    python run_pipeline.py --data path/to/dataset.csv --target label --sensitive gender race

    # Configure fairness/accuracy tradeoff:
    python run_pipeline.py --alpha 0.7 --beta 0.3

    # Disable Gemini LLM:
    python run_pipeline.py --no-gemini
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Phase 0: Data Loading
# ---------------------------------------------------------------------------

def create_synthetic_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic dataset with deliberate bias for demo/testing."""
    rng = np.random.RandomState(seed)

    gender = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    race = rng.choice(["group_A", "group_B", "group_C"], size=n, p=[0.6, 0.25, 0.15])

    age = rng.normal(35, 10, n).clip(18, 65).astype(int)
    income = rng.normal(50000, 15000, n).clip(15000, 150000)
    credit_score = rng.normal(650, 80, n).clip(300, 850).astype(int)

    # Proxy: zip_code correlated with race
    zip_base = {"group_A": 10000, "group_B": 20000, "group_C": 30000}
    zip_code = np.array([zip_base[r] + rng.randint(0, 100) for r in race])

    # Biased target
    approval_prob = 0.3 * np.ones(n)
    approval_prob[gender == "male"] += 0.2
    approval_prob[race == "group_A"] += 0.15
    approval_prob += (credit_score - 500) / 2000
    approval_prob += (income - 30000) / 200000
    approval_prob = np.clip(approval_prob, 0.05, 0.95)

    approved = (rng.random(n) < approval_prob).astype(int)

    return pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "zip_code": zip_code,
        "gender": gender,
        "race": race,
        "approved": approved,
    })


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from CSV."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


# ---------------------------------------------------------------------------
# Phase 1: Bias Detection (Built-in Analyzer)
# ---------------------------------------------------------------------------

def detect_bias(df: pd.DataFrame, target: str, sensitive_features: list) -> dict:
    """
    Phase 1: Automated bias detection.

    Analyzes the dataset for representation, outcome, fairness, and proxy bias,
    producing a structured report compatible with the mitigation engine.
    """
    report = {
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

    for feat in sensitive_features:
        if feat not in df.columns:
            print(f"  [WARN] Sensitive feature '{feat}' not in dataset, skipping.")
            continue

        # --- Distribution analysis ---
        props = df[feat].value_counts(normalize=True).to_dict()
        imbalance = max(props.values()) / max(min(props.values()), 1e-10)
        report["distribution_bias"][feat] = {
            "group_proportions": {str(k): round(float(v), 4) for k, v in props.items()},
            "imbalance_ratio": round(imbalance, 4),
        }

        if imbalance > 1.5:
            report["insights"].append(
                f"Representation imbalance in '{feat}': ratio = {imbalance:.2f}"
            )

        # --- Outcome analysis ---
        if target in df.columns:
            outcome_rates = df.groupby(feat)[target].mean().to_dict()
            disparity = max(outcome_rates.values()) - min(outcome_rates.values())
            report["outcome_bias"][feat] = {
                "outcome_rates": {str(k): round(float(v), 4) for k, v in outcome_rates.items()},
                "disparity": round(disparity, 4),
            }

            if disparity > 0.05:
                report["insights"].append(
                    f"Outcome disparity in '{feat}': gap = {disparity:.2%}"
                )

            # --- Fairness metrics ---
            rates = list(outcome_rates.values())
            dp_diff = max(rates) - min(rates)
            di_ratio = min(rates) / max(max(rates), 1e-10)
            report["fairness_metrics"][feat] = {
                "demographic_parity_difference": round(dp_diff, 4),
                "disparate_impact_ratio": round(di_ratio, 4),
            }

    # --- Proxy bias detection ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    for feat in sensitive_features:
        if feat not in df.columns:
            continue
        # Encode sensitive feature for correlation
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

    # --- Intersectional bias ---
    if len(sensitive_features) >= 2 and target in df.columns:
        try:
            combo_col = df[sensitive_features[0]].astype(str) + "_" + df[sensitive_features[1]].astype(str)
            combo_rates = df.groupby(combo_col)[target].mean()
            if combo_rates.max() - combo_rates.min() > 0.15:
                report["advanced_bias"]["intersectional_bias"] = {
                    "features": sensitive_features[:2],
                    "max_disparity": round(float(combo_rates.max() - combo_rates.min()), 4),
                    "group_rates": {str(k): round(float(v), 4) for k, v in combo_rates.items()},
                }
                report["insights"].append(
                    f"Intersectional bias detected across "
                    f"{sensitive_features[0]} x {sensitive_features[1]}"
                )
        except Exception:
            pass

    return report


# ---------------------------------------------------------------------------
# Phase 3: Artifact Export
# ---------------------------------------------------------------------------

def export_artifacts(result: dict, output_dir: str):
    """Export pipeline results to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Debiased dataset
    debiased_path = out / "debiased_dataset.csv"
    result["dataset_output"]["debiased_dataset"].to_csv(debiased_path, index=False)
    print(f"  [SAVED] Debiased dataset -> {debiased_path}")

    # 2. Full JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "bias_tags": result["bias_tags"],
        "best_strategy": result["ranking"]["best_strategy"],
        "best_score": result["ranking"]["best_score"],
        "ranking_table": result["ranking"]["ranking_table"],
        "accuracy_drop": result["model_output"]["accuracy_drop"],
        "fairness_improvement": result["dataset_output"]["fairness_improvement"],
        "bias_types_detected": result["dataset_output"]["bias_types_detected"],
        "llm_summary": result["llm_summary"],
        "comparison": result["dataset_output"].get("comparison", {}),
    }
    report_path = out / "mitigation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  [SAVED] Mitigation report  -> {report_path}")

    # 3. Best model (pickle)
    try:
        import pickle
        model = result["model_output"].get("best_model")
        if model is not None:
            model_path = out / "best_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"  [SAVED] Best model         -> {model_path}")
    except Exception as e:
        print(f"  [WARN] Could not save model: {e}")

    # 4. Bias report (from Phase 1)
    if "bias_report" in result:
        bias_path = out / "bias_report.json"
        with open(bias_path, "w") as f:
            json.dump(result["bias_report"], f, indent=2, default=str)
        print(f"  [SAVED] Bias report        -> {bias_path}")

    return out


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: str = None,
    target: str = "approved",
    sensitive_features: list = None,
    alpha: float = 0.6,
    beta: float = 0.4,
    model_type: str = "logistic_regression",
    gemini_enabled: bool = True,
    output_dir: str = "pipeline_output",
    gemini_api_key: str = None,
):
    """Run the complete AEGIS pipeline end-to-end."""

    start_time = time.time()

    print("=" * 70)
    print("  AEGIS -- Automated Bias Detection & Mitigation Pipeline")
    print("=" * 70)

    # --- Phase 0: Load Data ---
    print("\n[PHASE 0] Loading data...")
    if data_path:
        df = load_dataset(data_path)
    else:
        print("  No dataset specified -- generating synthetic biased data for demo")
        df = create_synthetic_dataset(n=2000)

    if sensitive_features is None:
        # Auto-detect: use categorical columns that aren't the target
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        sensitive_features = [c for c in cat_cols if c != target]
        if not sensitive_features:
            print("[ERROR] No sensitive features found. Specify with --sensitive.")
            sys.exit(1)

    print(f"  Dataset shape: {df.shape}")
    print(f"  Target: '{target}'")
    print(f"  Sensitive features: {sensitive_features}")
    print(f"  Target distribution:\n    {df[target].value_counts().to_dict()}")
    for feat in sensitive_features:
        print(f"  {feat} distribution: {df[feat].value_counts().to_dict()}")

    # --- Phase 1: Detect Bias ---
    print(f"\n[PHASE 1] Detecting bias...")
    bias_report = detect_bias(df, target, sensitive_features)
    print(f"  Insights found: {len(bias_report['insights'])}")
    for insight in bias_report["insights"]:
        print(f"    - {insight}")

    # --- Phase 2: Mitigate Bias ---
    print(f"\n[PHASE 2] Running mitigation engine...")
    print(f"  Config: alpha={alpha}, beta={beta}, model={model_type}")
    print(f"  Gemini: {'enabled' if gemini_enabled else 'disabled'}")

    # Set API key
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    elif "GEMINI_API_KEY" not in os.environ:
        gemini_enabled = False
        print("  [INFO] No GEMINI_API_KEY set -- Gemini disabled")

    from ErrorMitigation import BiasMitigationEngine

    engine = BiasMitigationEngine(config={
        "alpha": alpha,
        "beta": beta,
        "gemini_enabled": gemini_enabled,
        "gemini_model": "gemini-2.5-flash",
        "gemini_max_retries": 3,
    })

    result = engine.run(
        data=df,
        target=target,
        sensitive_features=sensitive_features,
        bias_report=bias_report,
        model_type=model_type,
    )

    # Attach the bias report for export
    result["bias_report"] = bias_report

    # --- Phase 3: Export Artifacts ---
    print(f"\n[PHASE 3] Exporting artifacts...")
    artifact_dir = export_artifacts(result, output_dir)

    # --- Summary ---
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
  Bias Types Detected:  {', '.join(result['dataset_output']['bias_types_detected'])}
  Best Strategy:        {result['ranking']['best_strategy']}
  Tradeoff Score:       {result['ranking']['best_score']:.4f}
  Accuracy Change:      {result['model_output']['accuracy_drop']:+.4f}
  Fairness Improvement: {result['dataset_output']['fairness_improvement']:.4f}
  Gemini Used:          {result['llm_summary']['gemini_used']}
  Output Directory:     {artifact_dir.resolve()}
  Time Elapsed:         {elapsed:.1f}s
""")

    # Print ranking table
    print("  Strategy Ranking:")
    for row in result["ranking"]["ranking_table"]:
        print(
            f"    #{row['rank']:2d}  {row['pipeline']:42s} "
            f"score={row['score']:.4f}  acc={row['accuracy']:.4f}  "
            f"dp_diff={row['demographic_parity_diff']:.4f}"
        )

    # Print LLM explanation
    llm = result["llm_summary"]
    if llm.get("gemini_used"):
        print(f"\n  Gemini Explanation:")
        print(f"  {'-' * 60}")
        if llm.get("bias_explanation"):
            print(f"  [Bias] {llm['bias_explanation'][:200]}")
        if llm.get("strategy_justification"):
            print(f"  [Strategy] {llm['strategy_justification'][:200]}")
        if llm.get("tradeoff_analysis"):
            print(f"  [Tradeoff] {llm['tradeoff_analysis'][:200]}")
        if llm.get("summary"):
            print(f"  [Summary] {llm['summary'][:300]}")
    else:
        print(f"\n  Fallback Summary:")
        print(f"    {result['llm_summary']['summary']}")

    print("\n" + "=" * 70)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AEGIS Automated Bias Detection & Mitigation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py --data dataset.csv --target income --sensitive gender race
  python run_pipeline.py --alpha 0.7 --beta 0.3 --model random_forest
  python run_pipeline.py --no-gemini --output results/
        """,
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to input CSV dataset (default: synthetic demo data)")
    parser.add_argument("--target", type=str, default="approved",
                        help="Target column name (default: approved)")
    parser.add_argument("--sensitive", nargs="+", default=None,
                        help="Sensitive feature column names (default: auto-detect)")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Accuracy weight in tradeoff score (default: 0.6)")
    parser.add_argument("--beta", type=float, default=0.4,
                        help="Fairness weight in tradeoff score (default: 0.4)")
    parser.add_argument("--model", type=str, default="logistic_regression",
                        choices=["logistic_regression", "random_forest", "xgboost"],
                        help="Model type for training (default: logistic_regression)")
    parser.add_argument("--no-gemini", action="store_true",
                        help="Disable Gemini LLM explanations")
    parser.add_argument("--gemini-key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--output", type=str, default="pipeline_output",
                        help="Output directory for artifacts (default: pipeline_output/)")

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        target=args.target,
        sensitive_features=args.sensitive,
        alpha=args.alpha,
        beta=args.beta,
        model_type=args.model,
        gemini_enabled=not args.no_gemini,
        output_dir=args.output,
        gemini_api_key=args.gemini_key,
    )


if __name__ == "__main__":
    main()
