"""
AEGIS — Central Configuration
================================

Single source of truth for all configurable parameters across the
entire AEGIS pipeline.  Values can be overridden via:

    1. Direct ``overrides`` dict passed to ``get_config()``
    2. Environment variables (prefixed with ``AEGIS_``)
    3. ``.env`` file in the project root
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    # Load .env file automatically
    env_path = Path(__file__).parent.parent.resolve() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "pipeline_output"
MODEL_DIR = PROJECT_ROOT / "model"

# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # -- Pipeline --
    "mode": "full_pipeline",  # "analysis" | "train" | "full_pipeline"

    # -- Bias classifier thresholds --
    "representation_imbalance_threshold": 0.2,
    "outcome_disparity_threshold": 0.1,
    "fairness_violation_threshold": 0.05,
    "proxy_correlation_threshold": 0.3,
    "intersectional_threshold": 0.15,
    "label_noise_threshold": 0.1,

    # -- Ranker weights --
    "alpha": 0.6,   # Weight for accuracy
    "beta": 0.4,    # Weight for fairness penalty

    # -- Training --
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "model_type": "logistic_regression",

    # -- SMOTE --
    "smote_k_neighbors": 5,

    # -- Threshold optimization --
    "threshold_search_steps": 50,

    # -- Gemini LLM --
    "gemini_model": "gemini-2.5-flash",
    "gemini_enabled": True,
    "gemini_temperature": 0.3,
    "gemini_max_tokens": 4096,
    "gemini_max_retries": 3,

    # -- Output --
    "output_dir": str(OUTPUT_DIR),
    "save_artifacts": True,
}


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level.

    Returns:
        Configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Config Builder
# ---------------------------------------------------------------------------

def get_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a merged configuration dictionary.

    Priority (highest → lowest):
        1. ``overrides`` dict
        2. Environment variables (``AEGIS_ALPHA``, ``AEGIS_BETA``, etc.)
        3. ``DEFAULT_CONFIG``

    Args:
        overrides: Key-value pairs to override defaults.

    Returns:
        Merged configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()

    # Read environment overrides
    env_map = {
        "AEGIS_ALPHA": ("alpha", float),
        "AEGIS_BETA": ("beta", float),
        "AEGIS_MODE": ("mode", str),
        "AEGIS_MODEL_TYPE": ("model_type", str),
        "AEGIS_GEMINI_ENABLED": ("gemini_enabled", lambda v: v.lower() in ("1", "true", "yes")),
        "AEGIS_GEMINI_MODEL": ("gemini_model", str),
        "GEMINI_API_KEY": (None, None),  # handled directly by the LLM client
    }

    for env_key, (cfg_key, cast_fn) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None and cfg_key is not None:
            try:
                config[cfg_key] = cast_fn(val)
            except (ValueError, TypeError):
                pass

    # Apply explicit overrides last
    if overrides:
        config.update(overrides)

    return config
