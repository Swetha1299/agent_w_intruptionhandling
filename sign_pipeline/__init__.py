"""Modular real-time sign-language assistant pipeline.

This package is intentionally isolated from existing application code.
"""

from .pipeline import run_sign_pipeline_real_time
from .stabilizer import TemporalSentenceStabilizer

__all__ = ["run_sign_pipeline_real_time", "TemporalSentenceStabilizer"]
