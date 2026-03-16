"""Sliding-window sign recognizer interfaces and placeholder implementation.

The placeholder recognizer is intentionally simple and designed to be replaced by
an ML model later without changing pipeline orchestration code.
"""

from __future__ import annotations

import logging
import importlib
import os
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .landmarks import LandmarkFrame

logger = logging.getLogger(__name__)


class BaseSignRecognizer(ABC):
    """Abstract recognizer interface for pluggable sign models."""

    @abstractmethod
    def predict_tokens(self, window: Sequence[LandmarkFrame]) -> List[str]:
        """Predict sign tokens from a sliding window of landmark frames."""


class MultiStreamPlaceholderRecognizer(BaseSignRecognizer):
    """Rule-based placeholder recognizer for prototype validation.

    Heuristics:
    - Both hands visible in majority of frames -> HELLO
    - Right wrist above right shoulder in majority -> HOW
    - Left wrist above left shoulder in majority -> YOU

    Replace this class with a real sequence model (e.g., Transformer/LSTM/TCN)
    later, keeping the same `predict_tokens` interface.
    """

    def __init__(self, min_votes: int = 5) -> None:
        self.min_votes = min_votes

    def _vote_frame(self, lf: LandmarkFrame) -> list[str]:
        votes: list[str] = []

        left_present = bool(np.any(lf.left_hand))
        right_present = bool(np.any(lf.right_hand))
        if left_present and right_present:
            votes.append("HELLO")

        # Pose index assumptions from MediaPipe Pose
        # left/right shoulders: 11,12 ; wrists: 15,16
        pose = lf.pose
        if pose.shape == (33, 4):
            left_wrist_y = pose[15, 1]
            right_wrist_y = pose[16, 1]
            left_shoulder_y = pose[11, 1]
            right_shoulder_y = pose[12, 1]

            if right_wrist_y < right_shoulder_y:
                votes.append("HOW")
            if left_wrist_y < left_shoulder_y:
                votes.append("YOU")

        return votes

    def predict_tokens(self, window: Sequence[LandmarkFrame]) -> List[str]:
        """Return a de-duplicated ordered token list from sliding-window votes."""
        if not window:
            return []

        all_votes: list[str] = []
        for frame in window:
            all_votes.extend(self._vote_frame(frame))

        if not all_votes:
            return []

        counts = Counter(all_votes)
        ordered = [token for token, n in counts.most_common() if n >= self.min_votes]

        logger.debug("Recognizer votes=%s ordered=%s", dict(counts), ordered)
        return ordered


class FusedTemporalTransformerRecognizer(BaseSignRecognizer):
    """Fused-landmark temporal recognizer (Transformer inference wrapper).

    Quick-prototyping behavior:
    - If an ONNX model file is configured and available, run model inference on
      fused landmarks (left hand + right hand + pose) over a temporal window.
    - If model runtime/model file is unavailable, seamlessly fall back to
      `MultiStreamPlaceholderRecognizer` to keep the app operational.

    Expected ONNX input shape: [B, T, D], where D = 258 by default.
    Expected ONNX output shape: [B, C] or [B, T, C].
    """

    def __init__(
        self,
        model_path: str | None = None,
        labels: Sequence[str] | None = None,
        window_size: int = 12,
        confidence_threshold: float = 0.55,
        fallback: BaseSignRecognizer | None = None,
    ) -> None:
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.labels = list(labels or ["HELLO", "HOW", "YOU", "THANK_YOU", "PLEASE"])
        self.fallback = fallback or MultiStreamPlaceholderRecognizer(min_votes=5)

        configured_path = model_path or os.getenv("SIGN_TRANSFORMER_ONNX_PATH", "").strip()
        self.model_path = configured_path if configured_path else None
        self._onnx_session = None
        self._onnx_input_name = None

        if self.model_path:
            self._load_onnx(self.model_path)
        else:
            logger.info("No transformer model path configured; using placeholder fallback recognizer")

    def _load_onnx(self, model_path: str) -> None:
        try:
            ort = importlib.import_module("onnxruntime")
        except Exception:
            logger.warning("onnxruntime is not installed; using placeholder fallback recognizer")
            return

        path = Path(model_path)
        if not path.exists():
            logger.warning("Transformer ONNX model not found at %s; using placeholder fallback recognizer", path)
            return

        try:
            self._onnx_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._onnx_input_name = self._onnx_session.get_inputs()[0].name
            logger.info("Loaded transformer recognizer model: %s", path)
        except Exception as exc:
            logger.warning("Failed to load transformer model (%s). Falling back. Error: %s", path, exc)
            self._onnx_session = None
            self._onnx_input_name = None

    @staticmethod
    def _fuse_frame(lf: LandmarkFrame) -> np.ndarray:
        """Convert one landmark frame to a single fused feature vector."""
        left = lf.left_hand.reshape(-1) if lf.left_hand.shape == (21, 3) else np.zeros(63, dtype=np.float32)
        right = lf.right_hand.reshape(-1) if lf.right_hand.shape == (21, 3) else np.zeros(63, dtype=np.float32)
        pose = lf.pose.reshape(-1) if lf.pose.shape == (33, 4) else np.zeros(132, dtype=np.float32)
        fused = np.concatenate([left, right, pose], axis=0).astype(np.float32)
        # D = 63 + 63 + 132 = 258
        return fused

    def _prepare_sequence(self, window: Sequence[LandmarkFrame]) -> np.ndarray:
        if not window:
            return np.zeros((1, self.window_size, 258), dtype=np.float32)

        features = [self._fuse_frame(frame) for frame in window[-self.window_size :]]
        if len(features) < self.window_size:
            pad_count = self.window_size - len(features)
            pad = [np.zeros(258, dtype=np.float32) for _ in range(pad_count)]
            features = pad + features

        seq = np.stack(features, axis=0)  # [T, D]
        return np.expand_dims(seq, axis=0)  # [1, T, D]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _predict_from_model(self, window: Sequence[LandmarkFrame]) -> List[str]:
        if self._onnx_session is None or self._onnx_input_name is None:
            return []

        seq = self._prepare_sequence(window)
        outputs = self._onnx_session.run(None, {self._onnx_input_name: seq})
        if not outputs:
            return []

        logits = outputs[0]
        if logits.ndim == 3:
            # [B, T, C] -> use last timestep
            logits = logits[:, -1, :]
        if logits.ndim != 2:
            return []

        probs = self._softmax(logits)
        pred_idx = int(np.argmax(probs[0]))
        confidence = float(probs[0, pred_idx])

        if confidence < self.confidence_threshold:
            return []
        if pred_idx < 0 or pred_idx >= len(self.labels):
            return []

        token = self.labels[pred_idx]
        if token in {"", "<blank>", "BLANK", "NONE", "PAD"}:
            return []

        return [token]

    def predict_tokens(self, window: Sequence[LandmarkFrame]) -> List[str]:
        """Predict tokens using transformer model, with placeholder fallback."""
        model_tokens = self._predict_from_model(window)
        if model_tokens:
            return model_tokens
        return self.fallback.predict_tokens(window)
