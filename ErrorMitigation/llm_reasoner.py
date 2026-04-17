"""
AEGIS -- LLM Reasoner (Gemini Integration)
============================================

Uses Google's Gemini model to generate human-readable explanations and
justifications for bias mitigation decisions.

**IMPORTANT**: The LLM does NOT make mitigation decisions.  All mitigation
logic is deterministic and handled by the core engine.  The LLM layer
is strictly for:

    1. Explaining what bias was detected
    2. Explaining why the chosen strategy makes sense
    3. Explaining fairness vs accuracy tradeoffs
    4. Summarising fairness improvements
    5. Generating user-friendly narrative summaries

The integration is modular and can be completely disabled via configuration
(``gemini_enabled = False``) without affecting the core pipeline.

Requires the ``GEMINI_API_KEY`` environment variable to be set.
"""

import os
from typing import Any, Dict, List, Optional

from .utils import get_config, get_logger, safe_import

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

ReasonerInput = Dict[str, Any]
# {
#     "bias_types": List[str],
#     "candidate_strategies": List[str],
#     "best_strategy": str,
#     "best_score": float,
#     "fairness_improvement": float,
#     "accuracy_drop": float,
#     "ranking_table": List[Dict],
#     "comparison": Dict[str, Any],
# }

ReasonerOutput = Dict[str, Any]
# {
#     "summary": str,
#     "bias_explanation": str,
#     "strategy_justification": str,
#     "tradeoff_analysis": str,
#     "recommendation": str,
#     "gemini_used": bool,
# }


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def _build_prompt(reasoner_input: ReasonerInput) -> str:
    """
    Build a structured prompt for Gemini based on the engine's outputs.

    Args:
        reasoner_input: Aggregated results from the engine.

    Returns:
        A detailed prompt string.
    """
    bias_types = reasoner_input.get("bias_types", [])
    candidates = reasoner_input.get("candidate_strategies", [])
    best = reasoner_input.get("best_strategy", "unknown")
    best_score = reasoner_input.get("best_score", 0.0)
    fairness_imp = reasoner_input.get("fairness_improvement", 0.0)
    acc_drop = reasoner_input.get("accuracy_drop", 0.0)
    ranking_table = reasoner_input.get("ranking_table", [])
    comparison = reasoner_input.get("comparison", {})

    # Build ranking summary
    ranking_lines = []
    for row in ranking_table[:5]:
        ranking_lines.append(
            f"  #{row.get('rank', '?')} {row.get('pipeline', '?')} -- "
            f"score: {row.get('score', 0):.4f}, "
            f"accuracy: {row.get('accuracy', 0):.4f}, "
            f"dp_diff: {row.get('demographic_parity_diff', 0):.4f}"
        )
    ranking_text = "\n".join(ranking_lines) if ranking_lines else "  (no ranking data)"

    prompt = f"""You are an AI fairness expert providing a concise, actionable explanation of bias mitigation results.

## Bias Detection Results
- **Bias types detected**: {', '.join(bias_types) if bias_types else 'None'}

## Mitigation Strategies Evaluated
- **Candidates tested**: {', '.join(candidates)}
- **Best strategy selected**: {best}
- **Best tradeoff score**: {best_score:.4f}

## Key Metrics
- **Fairness improvement** (reduction in unfairness): {fairness_imp:.4f}
- **Accuracy change**: {acc_drop:+.4f}

## Strategy Ranking (Top 5)
{ranking_text}

## Before vs After Comparison
{_format_comparison(comparison)}

---

Please provide a structured response with the following sections:

### 1. Bias Explanation
Explain what types of bias were detected and what they mean in plain language.

### 2. Strategy Justification
Explain why the selected strategy ({best}) is appropriate for the detected biases.

### 3. Tradeoff Analysis
Discuss the fairness vs accuracy tradeoff. Was the accuracy cost justified?

### 4. Summary
A 2-3 sentence executive summary suitable for a non-technical stakeholder.
"""
    return prompt


def _format_comparison(comparison: Dict[str, Any]) -> str:
    """Format comparison dict into readable text."""
    if not comparison:
        return "  (no comparison data available)"

    lines = []
    for key, value in comparison.items():
        if key.endswith("_improved"):
            continue  # Skip boolean flags in display
        clean_key = key.replace("_", " ").title()
        if isinstance(value, float):
            lines.append(f"  - {clean_key}: {value:.4f}")
        else:
            lines.append(f"  - {clean_key}: {value}")
    return "\n".join(lines) if lines else "  (empty)"


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

def _call_gemini(
    prompt: str,
    config: Dict[str, Any],
) -> Optional[str]:
    """
    Send a prompt to the Gemini API and return the response text.

    Tries the new ``google.genai`` SDK first, then falls back to the
    deprecated ``google.generativeai`` package.

    Args:
        prompt: The prompt string.
        config: Configuration dict with Gemini settings.

    Returns:
        Response text, or None if the call fails.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set -- skipping LLM reasoning.")
        return None

    model_name = config.get("gemini_model", "gemini-2.5-flash")

    # --- Try new google.genai SDK first ---
    genai_new = safe_import("google.genai")
    if genai_new is not None:
        max_retries = config.get("gemini_max_retries", 3)
        for attempt in range(max_retries):
            try:
                client = genai_new.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_new.types.GenerateContentConfig(
                        temperature=config.get("gemini_temperature", 0.3),
                        max_output_tokens=config.get("gemini_max_tokens", 1024),
                    ),
                )
                return response.text
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "503" in err_str or "UNAVAILABLE" in err_str
                        or "quota" in err_str.lower() or "rate" in err_str.lower()):
                    wait_time = 40 * (attempt + 1)
                    logger.warning(
                        f"Gemini rate limited (attempt {attempt+1}/{max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    import time
                    time.sleep(wait_time)
                else:
                    logger.warning(f"google.genai SDK failed: {e}")
                    break  # Non-retryable error, fall through

    # --- Fallback: legacy google.generativeai SDK ---
    genai_legacy = safe_import("google.generativeai")
    if genai_legacy is not None:
        try:
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai_legacy.types.GenerationConfig(
                    temperature=config.get("gemini_temperature", 0.3),
                    max_output_tokens=config.get("gemini_max_tokens", 1024),
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
            return None

    logger.warning("No Gemini SDK installed -- skipping LLM reasoning.")
    return None


# ---------------------------------------------------------------------------
# Fallback (No LLM)
# ---------------------------------------------------------------------------

def _generate_fallback_explanation(
    reasoner_input: ReasonerInput,
) -> ReasonerOutput:
    """
    Generate a deterministic, template-based explanation when Gemini
    is unavailable or disabled.

    Args:
        reasoner_input: Engine outputs.

    Returns:
        ReasonerOutput with template-based explanations.
    """
    bias_types = reasoner_input.get("bias_types", [])
    best = reasoner_input.get("best_strategy", "unknown")
    fairness_imp = reasoner_input.get("fairness_improvement", 0.0)
    acc_drop = reasoner_input.get("accuracy_drop", 0.0)

    bias_str = ", ".join(b.replace("_", " ") for b in bias_types) if bias_types else "none"
    direction = "improved" if fairness_imp > 0 else "did not improve"
    acc_note = (
        f"with a {abs(acc_drop):.2%} decrease in accuracy"
        if acc_drop > 0.005
        else "with negligible impact on accuracy"
    )

    bias_explanation = (
        f"The following bias types were detected in the dataset: {bias_str}. "
        "These indicate systematic differences in how the model treats different "
        "demographic groups."
    )

    strategy_justification = (
        f"The '{best.replace('_', ' ')}' strategy was selected as the best "
        f"approach to address the detected biases based on the fairness-accuracy "
        f"tradeoff score."
    )

    tradeoff_analysis = (
        f"Fairness {direction} by {abs(fairness_imp):.4f} {acc_note}. "
        "This represents an acceptable tradeoff for reducing algorithmic bias."
    )

    summary = (
        f"{bias_str.title()} {'were' if len(bias_types) > 1 else 'was'} detected. "
        f"The {best.replace('_', ' ')} strategy {direction} fairness "
        f"{acc_note}."
    )

    return {
        "summary": summary,
        "bias_explanation": bias_explanation,
        "strategy_justification": strategy_justification,
        "tradeoff_analysis": tradeoff_analysis,
        "recommendation": f"Apply the '{best}' mitigation strategy to the production pipeline.",
        "gemini_used": False,
    }


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def _parse_gemini_response(
    response_text: str,
    reasoner_input: ReasonerInput,
) -> ReasonerOutput:
    """
    Parse Gemini's response into structured sections.

    Args:
        response_text: Raw response from Gemini.
        reasoner_input: Original input (for context).

    Returns:
        ReasonerOutput with parsed sections.
    """
    sections = {
        "bias_explanation": "",
        "strategy_justification": "",
        "tradeoff_analysis": "",
        "summary": "",
    }

    # Section header detection patterns
    SECTION_MAP = [
        ("bias_explanation", ["bias explanation"]),
        ("strategy_justification", ["strategy justification", "strategy selection"]),
        ("tradeoff_analysis", ["tradeoff analysis", "trade-off analysis",
                               "tradeoff", "trade-off", "fairness vs accuracy",
                               "fairness versus accuracy"]),
        ("summary", ["summary", "executive summary", "conclusion"]),
    ]

    def _detect_section(line_text: str):
        """Check if a line is a section header."""
        stripped = line_text.strip().lower()
        # Remove markdown header markers
        clean = stripped.lstrip("#").strip().rstrip(":").strip()
        # Must look like a header (short, or starts with #)
        is_header_line = (
            line_text.strip().startswith("#") or
            (len(clean) < 60 and not clean.endswith("."))
        )
        if not is_header_line:
            return None
        for section_key, patterns in SECTION_MAP:
            for pattern in patterns:
                if pattern in clean:
                    return section_key
        return None

    current_section = None
    current_lines: List[str] = []

    for line in response_text.split("\n"):
        detected = _detect_section(line)
        if detected is not None:
            if current_section and current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = detected
            current_lines = []
        else:
            if current_section:
                current_lines.append(line)

    # Capture last section
    if current_section and current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    # Ensure summary is never empty
    if not sections["summary"]:
        # Build summary from other sections
        parts = []
        if sections["bias_explanation"]:
            # Take first sentence
            first = sections["bias_explanation"].split(".")[0].strip()
            if first:
                parts.append(first + ".")
        if sections["strategy_justification"]:
            first = sections["strategy_justification"].split(".")[0].strip()
            if first:
                parts.append(first + ".")
        if parts:
            sections["summary"] = " ".join(parts)
        else:
            # Use full response truncated
            sections["summary"] = response_text.strip()[:500]

    best = reasoner_input.get("best_strategy", "unknown")

    return {
        "summary": sections.get("summary", response_text[:500]),
        "bias_explanation": sections.get("bias_explanation", ""),
        "strategy_justification": sections.get("strategy_justification", ""),
        "tradeoff_analysis": sections.get("tradeoff_analysis", ""),
        "recommendation": f"Apply the '{best}' mitigation strategy to the production pipeline.",
        "gemini_used": True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_with_gemini(
    reasoner_input: ReasonerInput,
    config: Optional[Dict[str, Any]] = None,
) -> ReasonerOutput:
    """
    Generate an LLM-powered explanation of the mitigation results.

    If Gemini is disabled or unavailable, falls back to a deterministic
    template-based explanation.

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

    if not cfg.get("gemini_enabled", True):
        logger.info("Gemini reasoning disabled -- using fallback explanation.")
        return _generate_fallback_explanation(reasoner_input)

    logger.info("Generating LLM explanation via Gemini ...")

    prompt = _build_prompt(reasoner_input)
    response = _call_gemini(prompt, cfg)

    if response is None:
        logger.info("Gemini unavailable -- using fallback explanation.")
        return _generate_fallback_explanation(reasoner_input)

    parsed = _parse_gemini_response(response, reasoner_input)
    logger.info("Gemini explanation generated successfully.")
    return parsed
