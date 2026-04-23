"""
AEGIS — Unified Gemini Client
================================

Single Gemini integration point for the entire pipeline.  Replaces both
the old Groq-based ``llm_handler.py`` and the embedded Gemini calls in
``llm_reasoner.py``.

**IMPORTANT**: The LLM does NOT make mitigation decisions.  It is strictly
used for:
    1. Explaining detected bias
    2. Justifying the chosen mitigation strategy
    3. Analysing fairness vs accuracy tradeoffs
    4. Generating user-friendly summaries
    5. Detecting sensitive features from column names

Requires ``GEMINI_API_KEY`` environment variable.
"""

import os
import time
from typing import Any, Dict, List, Optional

from ..config import get_config, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Safe Import Helper
# ---------------------------------------------------------------------------

def _safe_import(module_name: str):
    """Attempt to import a module, returning None if unavailable."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Gemini Client Class
# ---------------------------------------------------------------------------

class GeminiClient:
    """
    Unified Gemini API client for all LLM operations in AEGIS.

    Uses ``google-genai`` (new SDK) first, then falls back to the
    legacy ``google-generativeai`` SDK.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = get_config(config)
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.enabled = bool(self.api_key) and self.config.get("gemini_enabled", True)

        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY not set — LLM explanations will use fallback templates."
            )

    # -----------------------------------------------------------------
    # Core API Call
    # -----------------------------------------------------------------

    def _call(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Gemini and return the response text.

        Args:
            prompt: The prompt string.

        Returns:
            Response text, or None if the call fails.
        """
        if not self.enabled:
            return None

        model_name = self.config.get("gemini_model", "gemini-2.5-flash")
        max_retries = self.config.get("gemini_max_retries", 3)

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
        except ImportError:
            logger.warning("langchain-google-genai not installed — skipping LLM call.")
            return None

        for attempt in range(max_retries):
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=self.config.get("gemini_temperature", 0.3),
                    max_output_tokens=self.config.get("gemini_max_tokens", 1024),
                )
                msg = HumanMessage(content=prompt)
                response = llm.invoke([msg])
                return response.content
            except Exception as e:
                err_str = str(e)
                if any(kw in err_str.lower() for kw in
                       ["429", "503", "unavailable", "quota", "rate"]):
                    wait_time = 40 * (attempt + 1)
                    logger.warning(
                        f"Gemini rate limited (attempt {attempt+1}/{max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LangChain Gemini API call failed: {e}")
                    break

        return None

    # -----------------------------------------------------------------
    # Public Methods
    # -----------------------------------------------------------------

    def detect_sensitive_features(self, columns: List[str]) -> List[str]:
        """
        Use Gemini to identify sensitive/protected attribute columns.

        Args:
            columns: List of column names from the dataset.

        Returns:
            List of column names identified as sensitive features.
        """
        prompt = f"""You are an AI fairness expert.

Given these dataset columns:
{columns}

Identify columns that represent sensitive/protected attributes that could
cause discrimination or bias (e.g. gender, race, age, religion, nationality).

Return ONLY a Python list of column names, nothing else.
Example: ["gender", "race"]
"""
        response = self._call(prompt)

        if response:
            try:
                import ast
                # Extract list from response
                text = response.strip()
                if "[" in text and "]" in text:
                    start = text.index("[")
                    end = text.rindex("]") + 1
                    result = ast.literal_eval(text[start:end])
                    # Filter to only columns that actually exist
                    return [c for c in result if c in columns]
            except Exception:
                pass

        # Fallback: heuristic detection
        sensitive_keywords = [
            "sex", "gender", "race", "ethnicity", "religion", "age",
            "nationality", "native_country", "marital", "disability",
        ]
        return [
            col for col in columns
            if any(kw in col.lower() for kw in sensitive_keywords)
        ]

    def explain_bias(self, bias_report: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of detected bias.

        Args:
            bias_report: Structured bias report from the detection phase.

        Returns:
            Explanation string.
        """
        prompt = f"""You are an AI fairness expert. Analyse these bias detection results
and provide a clear, actionable explanation.

Bias Report:
{_format_dict(bias_report)}

Explain:
1. What types of bias were detected and what they mean
2. Which demographic groups are most affected
3. Why this matters for the system's fairness
4. Brief recommendation for mitigation

Keep the response concise (under 300 words).
"""
        response = self._call(prompt)
        if response:
            return response

        # Fallback
        insights = bias_report.get("insights", [])
        if insights:
            return "Bias detected: " + "; ".join(insights)
        return "Bias analysis complete. See metrics for details."

    def explain_mitigation(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate structured explanations of mitigation results.

        This is the main explanation method used after Phase 2 completes.

        Args:
            context: Dict with keys like bias_types, best_strategy,
                fairness_improvement, accuracy_drop, ranking_table, comparison.

        Returns:
            Dict with keys: summary, bias_explanation, strategy_justification,
            tradeoff_analysis, recommendation, gemini_used.
        """
        prompt = _build_mitigation_prompt(context)
        response = self._call(prompt)

        if response:
            parsed = _parse_structured_response(response, context)
            parsed["gemini_used"] = True
            return parsed

        return _generate_fallback_explanation(context)

    def explain_tradeoff(self, comparison: Dict[str, Any]) -> str:
        """
        Explain the fairness vs accuracy tradeoff.

        Args:
            comparison: Before/after metrics comparison.

        Returns:
            Explanation string.
        """
        prompt = f"""You are an AI fairness expert. Analyse this fairness vs accuracy tradeoff:

{_format_dict(comparison)}

Discuss:
1. Was the accuracy cost justified by fairness gains?
2. Which groups benefited most?
3. Are there any remaining concerns?

Keep the response under 200 words.
"""
        response = self._call(prompt)
        return response or "Tradeoff analysis unavailable — see metrics."


# ---------------------------------------------------------------------------
# Module-Level Convenience Function
# ---------------------------------------------------------------------------

def generate_explanation(context: dict, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Quick convenience function to generate an LLM explanation.

    Args:
        context: Context dict to explain.
        config: Optional config overrides.

    Returns:
        Explanation string.
    """
    client = GeminiClient(config=config)
    return client.explain_bias(context)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _format_dict(d: Dict[str, Any], max_depth: int = 3) -> str:
    """Format a dict for prompt inclusion, truncating deep nesting."""
    import json
    try:
        return json.dumps(d, indent=2, default=str)[:3000]
    except Exception:
        return str(d)[:3000]


def _build_mitigation_prompt(context: Dict[str, Any]) -> str:
    """Build a structured prompt for mitigation explanation."""
    bias_types = context.get("bias_types", [])
    candidates = context.get("candidate_strategies", [])
    best = context.get("best_strategy", "unknown")
    best_score = context.get("best_score", 0.0)
    fairness_imp = context.get("fairness_improvement", 0.0)
    acc_drop = context.get("accuracy_drop", 0.0)
    ranking_table = context.get("ranking_table", [])

    ranking_lines = []
    for row in ranking_table[:5]:
        ranking_lines.append(
            f"  #{row.get('rank', '?')} {row.get('pipeline', '?')} — "
            f"score: {row.get('score', 0):.4f}, "
            f"accuracy: {row.get('accuracy', 0):.4f}, "
            f"dp_diff: {row.get('demographic_parity_diff', 0):.4f}"
        )
    ranking_text = "\n".join(ranking_lines) if ranking_lines else "  (no data)"

    return f"""You are an AI fairness expert providing a concise, actionable explanation.

## Bias Detection Results
- **Bias types detected**: {', '.join(bias_types) if bias_types else 'None'}

## Mitigation Strategies Evaluated
- **Candidates tested**: {', '.join(candidates)}
- **Best strategy selected**: {best}
- **Best tradeoff score**: {best_score:.4f}

## Key Metrics
- **Fairness improvement**: {fairness_imp:.4f}
- **Accuracy change**: {acc_drop:+.4f}

## Strategy Ranking (Top 5)
{ranking_text}

Provide a structured response with these sections:

### 1. Bias Explanation
What was detected and what it means.

### 2. Strategy Justification
Why {best} is appropriate for these biases.

### 3. Tradeoff Analysis
Was the accuracy cost justified?

### 4. Summary
2-3 sentence executive summary for a non-technical stakeholder.
"""


def _parse_structured_response(
    response_text: str,
    context: Dict[str, Any],
) -> Dict[str, str]:
    """Parse Gemini's structured response into sections."""
    sections = {
        "bias_explanation": "",
        "strategy_justification": "",
        "tradeoff_analysis": "",
        "summary": "",
    }

    SECTION_MAP = [
        ("bias_explanation", ["bias explanation"]),
        ("strategy_justification", ["strategy justification", "strategy selection"]),
        ("tradeoff_analysis", ["tradeoff analysis", "trade-off analysis",
                               "tradeoff", "trade-off"]),
        ("summary", ["summary", "executive summary", "conclusion"]),
    ]

    def _detect_section(line_text: str):
        stripped = line_text.strip().lower()
        clean = stripped.lstrip("#").strip().rstrip(":").strip()
        is_header = (
            line_text.strip().startswith("#") or
            (len(clean) < 60 and not clean.endswith("."))
        )
        if not is_header:
            return None
        for section_key, patterns in SECTION_MAP:
            for pattern in patterns:
                if pattern in clean:
                    return section_key
        return None

    current_section = None
    current_lines = []

    for line in response_text.split("\n"):
        detected = _detect_section(line)
        if detected is not None:
            if current_section and current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = detected
            current_lines = []
        elif current_section:
            current_lines.append(line)

    if current_section and current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    if not sections["summary"]:
        parts = []
        for key in ["bias_explanation", "strategy_justification"]:
            if sections[key]:
                first = sections[key].split(".")[0].strip()
                if first:
                    parts.append(first + ".")
        sections["summary"] = " ".join(parts) if parts else response_text[:500]

    best = context.get("best_strategy", "unknown")
    return {
        "summary": sections.get("summary", ""),
        "bias_explanation": sections.get("bias_explanation", ""),
        "strategy_justification": sections.get("strategy_justification", ""),
        "tradeoff_analysis": sections.get("tradeoff_analysis", ""),
        "recommendation": f"Apply the '{best}' mitigation strategy.",
        "gemini_used": True,
    }


def _generate_fallback_explanation(context: Dict[str, Any]) -> Dict[str, str]:
    """Generate a template-based explanation when Gemini is unavailable."""
    bias_types = context.get("bias_types", [])
    best = context.get("best_strategy", "unknown")
    fairness_imp = context.get("fairness_improvement", 0.0)
    acc_drop = context.get("accuracy_drop", 0.0)

    bias_str = ", ".join(b.replace("_", " ") for b in bias_types) if bias_types else "none"
    direction = "improved" if fairness_imp > 0 else "did not improve"
    acc_note = (
        f"with a {abs(acc_drop):.2%} decrease in accuracy"
        if acc_drop > 0.005
        else "with negligible impact on accuracy"
    )

    return {
        "summary": (
            f"{bias_str.title()} {'were' if len(bias_types) > 1 else 'was'} detected. "
            f"The {best.replace('_', ' ')} strategy {direction} fairness {acc_note}."
        ),
        "bias_explanation": (
            f"The following bias types were detected: {bias_str}. These indicate "
            "systematic differences in how different demographic groups are treated."
        ),
        "strategy_justification": (
            f"The '{best.replace('_', ' ')}' strategy was selected as the best approach "
            "based on the fairness-accuracy tradeoff score."
        ),
        "tradeoff_analysis": (
            f"Fairness {direction} by {abs(fairness_imp):.4f} {acc_note}. "
            "This represents an acceptable tradeoff for reducing algorithmic bias."
        ),
        "recommendation": f"Apply the '{best}' mitigation strategy.",
        "gemini_used": False,
    }
