"""
AEGIS — CLI Entry Point
==========================

Command-line interface for the AEGIS AI Bias Governance Engine.

Usage::

    # Full pipeline with synthetic data (demo):
    python run_aegis.py

    # Analysis mode only:
    python run_aegis.py --mode analysis

    # Full pipeline with custom CSV:
    python run_aegis.py --data dataset.csv --target income --sensitive gender race

    # Training mode with custom weights:
    python run_aegis.py --mode train --alpha 0.7 --beta 0.3 --model-type random_forest

    # Disable Gemini LLM:
    python run_aegis.py --no-gemini

    # Specify Gemini API key:
    python run_aegis.py --gemini-key YOUR_KEY
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="AEGIS — AI Bias Governance Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_aegis.py
  python run_aegis.py --mode analysis
  python run_aegis.py --data dataset.csv --target income --sensitive gender race
  python run_aegis.py --mode train --alpha 0.7 --beta 0.3
  python run_aegis.py --no-gemini
        """,
    )

    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to input CSV dataset (default: synthetic demo data)"
    )
    parser.add_argument(
        "--mode", type=str, default="full_pipeline",
        choices=["analysis", "train", "full_pipeline"],
        help="Pipeline mode (default: full_pipeline)"
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Target column name (default: auto-detect)"
    )
    parser.add_argument(
        "--sensitive", nargs="+", default=None,
        help="Sensitive feature column names (default: auto-detect)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.6,
        help="Accuracy weight in tradeoff score (default: 0.6)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.4,
        help="Fairness weight in tradeoff score (default: 0.4)"
    )
    parser.add_argument(
        "--model-type", type=str, default="logistic_regression",
        choices=["logistic_regression", "random_forest", "xgboost"],
        help="Model type for training (default: logistic_regression)"
    )
    parser.add_argument(
        "--no-gemini", action="store_true",
        help="Disable Gemini LLM explanations"
    )
    parser.add_argument(
        "--gemini-key", type=str, default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--output", type=str, default="pipeline_output",
        help="Output directory for artifacts (default: pipeline_output/)"
    )

    args = parser.parse_args()

    # Set API key if provided
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key

    # Build input data
    input_data = {
        "dataset": args.data,  # None = synthetic
        "mode": args.mode,
        "model": None,
        "target": args.target,
        "sensitive_features": args.sensitive,
        "config": {
            "alpha": args.alpha,
            "beta": args.beta,
            "model_type": args.model_type,
            "gemini_enabled": not args.no_gemini,
            "output_dir": args.output,
        },
    }

    # Run pipeline
    from core import run_pipeline
    result = run_pipeline(input_data)

    return result


if __name__ == "__main__":
    main()
