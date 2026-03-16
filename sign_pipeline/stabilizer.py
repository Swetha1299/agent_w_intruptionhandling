"""Temporal stabilization and sentence-boundary detection for sign recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .landmarks import LandmarkFrame
from .text_converter import SignTextConverter


@dataclass
class StabilizerResult:
    """Output of one stabilizer update step."""

    live_tokens: tuple[str, ...]
    live_text: str
    settled_tokens: tuple[str, ...]
    settled_text: str
    is_settled: bool
    motion_score: float


class TemporalSentenceStabilizer:
    """Stabilize rolling token predictions into committed sentences.

    Strategy:
    - wait for the same token tuple across multiple windows
    - watch for low-motion / empty-token pause
    - commit a settled sentence only after a pause boundary
    """

    def __init__(
        self,
        converter: SignTextConverter,
        stability_windows: int = 3,
        pause_frames: int = 6,
        motion_threshold: float = 0.006,
    ) -> None:
        self.converter = converter
        self.stability_windows = stability_windows
        self.pause_frames = pause_frames
        self.motion_threshold = motion_threshold

        self._prev_features: np.ndarray | None = None
        self._candidate_tokens: tuple[str, ...] = ()
        self._candidate_count = 0
        self._active_tokens: tuple[str, ...] = ()
        self._low_motion_frames = 0
        self._empty_frames = 0
        self._last_committed_tokens: tuple[str, ...] = ()
        self._ready_for_repeat = True

    def _frame_features(self, frame: LandmarkFrame) -> np.ndarray:
        left = frame.left_hand.reshape(-1) if frame.left_hand.shape == (21, 3) else np.zeros(63, dtype=np.float32)
        right = frame.right_hand.reshape(-1) if frame.right_hand.shape == (21, 3) else np.zeros(63, dtype=np.float32)
        pose = frame.pose.reshape(-1) if frame.pose.shape == (33, 4) else np.zeros(132, dtype=np.float32)
        return np.concatenate([left, right, pose], axis=0).astype(np.float32)

    def update(self, tokens: Sequence[str], frame: LandmarkFrame) -> StabilizerResult:
        token_tuple = tuple(tokens)
        features = self._frame_features(frame)

        motion_score = 0.0
        if self._prev_features is not None and self._prev_features.shape == features.shape:
            motion_score = float(np.mean(np.abs(features - self._prev_features)))
        self._prev_features = features

        low_motion = motion_score < self.motion_threshold
        self._low_motion_frames = self._low_motion_frames + 1 if low_motion else 0

        if token_tuple:
            self._empty_frames = 0
            if token_tuple == self._candidate_tokens:
                self._candidate_count += 1
            else:
                self._candidate_tokens = token_tuple
                self._candidate_count = 1

            if self._candidate_count >= self.stability_windows:
                self._active_tokens = token_tuple
        else:
            self._empty_frames += 1
            self._candidate_count = 0
            self._candidate_tokens = ()

        if self._empty_frames >= self.pause_frames:
            self._ready_for_repeat = True

        settled_tokens: tuple[str, ...] = ()
        settled_text = ""
        is_settled = False

        should_commit = bool(self._active_tokens) and (
            self._empty_frames >= self.pause_frames or self._low_motion_frames >= self.pause_frames
        )
        can_commit = self._ready_for_repeat or self._active_tokens != self._last_committed_tokens

        if should_commit and can_commit:
            settled_tokens = self._active_tokens
            settled_text = self.converter.to_text(settled_tokens)
            is_settled = bool(settled_text)
            if is_settled:
                self._last_committed_tokens = settled_tokens
                self._ready_for_repeat = False
            self._active_tokens = ()
            self._candidate_tokens = ()
            self._candidate_count = 0

        live_tokens = self._active_tokens or self._candidate_tokens or token_tuple
        live_text = self.converter.to_text(live_tokens)

        return StabilizerResult(
            live_tokens=live_tokens,
            live_text=live_text,
            settled_tokens=settled_tokens,
            settled_text=settled_text,
            is_settled=is_settled,
            motion_score=motion_score,
        )
