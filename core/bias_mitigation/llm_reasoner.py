"""
AEGIS — LLM Reasoner (Gemini Integration)
============================================

Wrapper that delegates to the unified Gemini client at ``core.llm.gemini_client``.

**IMPORTANT**: The LLM does NOT make mitigation decisions. All mitigation
logic is deterministic and handled by the core engine. The LLM layer
is strictly for explaining results.
"""

from typing import Any, Dict, Optional

from ..config import get_config, get_logger
from ..llm.gemini_client import GeminiClient

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

ReasonerInput = Dict[str, Any]
ReasonerOutput = Dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_with_gemini(
    reasoner_input: ReasonerInput,
    config: Optional[Dict[str, Any]] = None,
) -> ReasonerOutput:
    """
    Generate an LLM-powered explanation of mitigation results.

    Delegates to the unified GeminiClient. If Gemini is disabled or
    unavailable, falls back to template-based explanations.

    Args:
        reasoner_input: Aggregated results from the engine containing:
            - bias_types
            - candidate_strategies
            - best_strategy
            - best_score
            - fairness_improvement
            - accuracy_drop
            - ranking_table
            - comparison

        config: Optional configuration overrides.

    Returns:
        ReasonerOutput with summary, explanations, and justifications.
    """
    cfg = get_config(config)
    client = GeminiClient(config=cfg)
    return client.explain_mitigation(reasoner_input)
