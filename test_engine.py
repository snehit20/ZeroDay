"""
AEGIS — Test Script
=====================

End-to-end test that creates a synthetic biased dataset, constructs a
sample bias report, and runs the full mitigation pipeline.

Usage:
    python test_engine.py
"""

import numpy as np
import pandas as pd


def create_synthetic_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic dataset with deliberate bias for testing.

    The dataset simulates a loan-approval scenario with:
        - gender bias (males approved more often)
        - race bias (group A approved more often)
        - proxy feature (zip_code correlated with race)
    """
    rng = np.random.RandomState(seed)

    gender = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    race = rng.choice(["group_A", "group_B", "group_C"], size=n, p=[0.6, 0.25, 0.15])

    age = rng.normal(35, 10, n).clip(18, 65).astype(int)
    income = rng.normal(50000, 15000, n).clip(15000, 150000)
    credit_score = rng.normal(650, 80, n).clip(300, 850).astype(int)

    # Proxy feature: zip_code is correlated with race
    zip_base = {"group_A": 10000, "group_B": 20000, "group_C": 30000}
    zip_code = np.array([zip_base[r] + rng.randint(0, 100) for r in race])

    # Biased target: males and group_A get approved more
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


def create_sample_bias_report(df: pd.DataFrame) -> dict:
    """
    Create a sample bias report simulating Phase 1 output.
    """
    # Distribution bias
    gender_props = df["gender"].value_counts(normalize=True).to_dict()
    race_props = df["race"].value_counts(normalize=True).to_dict()

    # Outcome bias
    gender_outcomes = df.groupby("gender")["approved"].mean().to_dict()
    race_outcomes = df.groupby("race")["approved"].mean().to_dict()

    return {
        "distribution_bias": {
            "gender": {
                "group_proportions": {str(k): float(v) for k, v in gender_props.items()},
                "imbalance_ratio": max(gender_props.values()) / max(min(gender_props.values()), 1e-10),
            },
            "race": {
                "group_proportions": {str(k): float(v) for k, v in race_props.items()},
                "imbalance_ratio": max(race_props.values()) / max(min(race_props.values()), 1e-10),
            },
        },
        "outcome_bias": {
            "gender": {
                "outcome_rates": {str(k): float(v) for k, v in gender_outcomes.items()},
                "disparity": max(gender_outcomes.values()) - min(gender_outcomes.values()),
            },
            "race": {
                "outcome_rates": {str(k): float(v) for k, v in race_outcomes.items()},
                "disparity": max(race_outcomes.values()) - min(race_outcomes.values()),
            },
        },
        "fairness_metrics": {
            "gender": {
                "demographic_parity_difference": abs(
                    gender_outcomes.get("male", 0) - gender_outcomes.get("female", 0)
                ),
                "disparate_impact_ratio": min(gender_outcomes.values()) / max(
                    max(gender_outcomes.values()), 1e-10
                ),
            },
            "race": {
                "demographic_parity_difference": max(race_outcomes.values()) - min(
                    race_outcomes.values()
                ),
            },
        },
        "advanced_bias": {
            "proxy_bias": {
                "zip_code": {
                    "correlation": 0.72,
                    "correlated_with": "race",
                },
            },
            "intersectional_bias": {},
            "label_bias": {},
        },
        "insights": [
            "Significant gender imbalance detected (65% male vs 35% female).",
            "Approval rate gap between genders exceeds 10%.",
            "zip_code is a strong proxy for race.",
        ],
    }


def main():
    print("=" * 70)
    print("AEGIS Bias Mitigation Engine — End-to-End Test")
    print("=" * 70)

    # Create synthetic dataset
    print("\n[1] Creating synthetic dataset …")
    df = create_synthetic_dataset(n=1500)
    print(f"    Dataset shape: {df.shape}")
    print(f"    Target distribution:\n{df['approved'].value_counts().to_string()}")
    print(f"    Gender distribution:\n{df['gender'].value_counts().to_string()}")
    print(f"    Race distribution:\n{df['race'].value_counts().to_string()}")

    # Create bias report
    print("\n[2] Creating sample bias report …")
    bias_report = create_sample_bias_report(df)
    print(f"    Report sections: {list(bias_report.keys())}")

    # Run engine
    print("\n[3] Running mitigation engine …")
    from ErrorMitigation import BiasMitigationEngine

    # Set Gemini API key from environment
    import os
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")


    engine = BiasMitigationEngine(config={
        "alpha": 0.6,
        "beta": 0.4,
        "gemini_enabled": True,
    })

    result = engine.run(
        data=df,
        target="approved",
        sensitive_features=["gender", "race"],
        bias_report=bias_report,
        model_type="logistic_regression",
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Bias tags
    print("\n[A] Bias Tags:")
    for tag, detected in result["bias_tags"].items():
        status = "[x] DETECTED" if detected else "[ ] not detected"
        print(f"    {tag:30s} {status}")

    # Ranking
    print("\n[B] Strategy Ranking:")
    for row in result["ranking"]["ranking_table"]:
        print(
            f"    #{row['rank']} {row['pipeline']:40s} "
            f"score={row['score']:.4f}  acc={row['accuracy']:.4f}  "
            f"dp_diff={row['demographic_parity_diff']:.4f}"
        )

    # Best strategy
    print(f"\n[C] Best Strategy: {result['ranking']['best_strategy']}")
    print(f"    Best Score: {result['ranking']['best_score']:.4f}")

    # Model output summary
    mo = result["model_output"]
    print(f"\n[D] Accuracy Drop: {mo['accuracy_drop']:.4f}")
    print(f"    Strategy Reason: {mo['strategy_reason'][:100]}…")

    # Dataset output summary
    do = result["dataset_output"]
    print(f"\n[E] Debiased Dataset Shape: {do['debiased_dataset'].shape}")
    print(f"    Fairness Improvement: {do['fairness_improvement']:.4f}")
    print(f"    Bias Types Detected: {do['bias_types_detected']}")

    # LLM Summary
    print(f"\n[F] LLM Summary:")
    print(f"    {result['llm_summary']['summary']}")
    print(f"    Gemini Used: {result['llm_summary']['gemini_used']}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE [OK]")
    print("=" * 70)


if __name__ == "__main__":
    main()
