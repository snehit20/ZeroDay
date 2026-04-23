"""
AEGIS — Bias Detection Module
"""

from .detector import detect_bias
from .preprocessing import preprocess_dataset

__all__ = ["detect_bias", "preprocess_dataset"]
