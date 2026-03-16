"""MediaPipe Tasks landmark extraction for hands, pose, and face streams."""

from __future__ import annotations

import importlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LandmarkFrame:
    """Structured multi-stream landmarks for one frame."""

    left_hand: np.ndarray  # shape: (21, 3)
    right_hand: np.ndarray  # shape: (21, 3)
    pose: np.ndarray  # shape: (33, 4) -> x,y,z,visibility
    face: np.ndarray  # shape: (468, 3)


def _zeros(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


DEFAULT_TASK_URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


class MultiStreamLandmarkExtractor:
    """Extract hands, pose, and face landmarks via MediaPipe Tasks.

    Uses landmarker `.task` models (auto-downloaded by default) and returns
    fixed-shape arrays suitable for downstream sequence models.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_face: bool = True,
    ) -> None:
        try:
            mp = importlib.import_module("mediapipe")
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for sign landmark extraction. Install with: pip install mediapipe"
            ) from exc

        self._mp = mp
        self._hand = None
        self._pose = None
        self._face = None
        self._enable_face = enable_face

        models_dir = Path(__file__).resolve().parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        hand_task = self._resolve_task_path("SIGN_PIPELINE_HAND_TASK_PATH", models_dir / "hand_landmarker.task", DEFAULT_TASK_URLS["hand"])
        pose_task = self._resolve_task_path("SIGN_PIPELINE_POSE_TASK_PATH", models_dir / "pose_landmarker.task", DEFAULT_TASK_URLS["pose"])
        face_task = ""
        if self._enable_face:
            face_task = self._resolve_task_path(
                "SIGN_PIPELINE_FACE_TASK_PATH",
                models_dir / "face_landmarker.task",
                DEFAULT_TASK_URLS["face"],
            )

        self._init_tasks(mp, hand_task, pose_task, face_task, min_detection_confidence, min_tracking_confidence)

    def _resolve_task_path(self, env_var: str, default_path: Path, download_url: str) -> str:
        env_path = os.getenv(env_var)
        path = Path(env_path) if env_path else default_path

        if not path.exists():
            logger.info("Downloading %s model to %s", env_var, path)
            try:
                urllib.request.urlretrieve(download_url, str(path))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download model for {env_var}. Set {env_var} to a valid .task file path."
                ) from exc
        return str(path)

    def _init_tasks(
        self,
        mp,
        hand_task: str,
        pose_task: str,
        face_task: str,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        self._hand = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_task),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

        self._pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_task),
                running_mode=VisionRunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

        if self._enable_face:
            self._face = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=face_task),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_tracking_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            )

    def close(self) -> None:
        """Close MediaPipe resources."""
        if self._hand is not None:
            self._hand.close()
        if self._pose is not None:
            self._pose.close()
        if self._face is not None:
            self._face.close()

    def _extract_hands(self, hand_result) -> tuple[np.ndarray, np.ndarray]:
        left = _zeros((21, 3))
        right = _zeros((21, 3))

        hand_landmarks = getattr(hand_result, "hand_landmarks", []) or []
        handedness = getattr(hand_result, "handedness", []) or []

        for lm_list, handedness_list in zip(hand_landmarks, handedness):
            if not lm_list or not handedness_list:
                continue
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
            if arr.shape != (21, 3):
                continue

            label = getattr(handedness_list[0], "category_name", "") if handedness_list else ""
            if label == "Left":
                left = arr
            elif label == "Right":
                right = arr

        return left, right

    def _extract_pose(self, pose_result) -> np.ndarray:
        pose_landmarks = getattr(pose_result, "pose_landmarks", []) or []
        if not pose_landmarks:
            return _zeros((33, 4))

        lms = pose_landmarks[0]
        arr = np.array(
            [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)] for lm in lms],
            dtype=np.float32,
        )
        return arr if arr.shape == (33, 4) else _zeros((33, 4))

    def _extract_face(self, face_result) -> np.ndarray:
        face_landmarks = getattr(face_result, "face_landmarks", []) or []
        if not face_landmarks:
            return _zeros((468, 3))

        # Face task may return >468 points in some variants. Keep first 468.
        lms = face_landmarks[0][:468]
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        if arr.shape[0] < 468:
            padded = _zeros((468, 3))
            padded[: arr.shape[0], :] = arr
            return padded
        return arr if arr.shape == (468, 3) else _zeros((468, 3))

    def extract(self, frame_bgr: np.ndarray) -> LandmarkFrame:
        """Extract and return structured landmarks for a BGR frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)

        hand_result = self._hand.detect(mp_image)
        pose_result = self._pose.detect(mp_image)
        face_result = self._face.detect(mp_image) if self._face is not None else None

        left_hand, right_hand = self._extract_hands(hand_result)
        pose = self._extract_pose(pose_result)
        face = self._extract_face(face_result)

        return LandmarkFrame(
            left_hand=left_hand,
            right_hand=right_hand,
            pose=pose,
            face=face,
        )
