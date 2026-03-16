"""Camera capture utilities for real-time frame streaming."""

from __future__ import annotations

import logging
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """OpenCV webcam capture stream.

    Args:
        camera_index: Camera device index.
        width: Optional target frame width.
        height: Optional target frame height.
    """

    def __init__(self, camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the webcam device."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.camera_index}")

        if self.width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        logger.info("Camera opened: index=%s", self.camera_index)

    def close(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames in real-time until capture fails."""
        if self._cap is None:
            self.open()

        assert self._cap is not None
        while True:
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Camera frame read failed")
                break
            yield frame

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
