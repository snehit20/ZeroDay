"""
AEGIS — AI Bias Governance Engine
====================================

Production-grade AI pipeline for bias detection, mitigation,
dataset comparison, and model explainability.

Usage::

    from core.pipeline import run_pipeline

    result = run_pipeline({
        "dataset": "path/to/data.csv",
        "mode": "full_pipeline",          # or "analysis" or "train"
        "model": None,                    # optional pre-trained model
    })
"""

__version__ = "2.0.0"
__author__ = "AEGIS Team"

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
