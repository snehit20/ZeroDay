"""
AEGIS — Bias Mitigation Engine
================================

Production-grade orchestration engine for bias mitigation in ML systems.

Phase 2 of the AEGIS pipeline:
    Phase 1 -> Bias Detection
    Phase 2 -> Bias Mitigation (this module)
    Phase 3 -> Dataset Comparison
    Phase 4 -> Model Explainability & Comparison

Modules:
    - classifier: Converts raw bias reports into structured bias tags
    - selector: Maps bias tags to relevant mitigation strategies
    - generator: Generates candidate mitigation pipelines
    - strategies: Implements all mitigation techniques
    - trainer: Trains models for each mitigation pipeline
    - evaluator: Computes performance and fairness metrics
    - ranker: Ranks candidate strategies by fairness-accuracy tradeoff
    - engine: Main orchestrator tying the full pipeline together
    - utils: Shared utility functions
"""

__version__ = "2.0.0"
__author__ = "AEGIS Team"

from .engine import BiasMitigationEngine

__all__ = ["BiasMitigationEngine"]
