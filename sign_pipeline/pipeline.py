"""End-to-end real-time sign-language assistant orchestration pipeline."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from typing import Deque, Dict, Generator, Optional

import cv2

from .camera import CameraStream
from .gemini_client import GeminiNativeAudioClient
from .landmarks import LandmarkFrame, MultiStreamLandmarkExtractor
from .recognizer import BaseSignRecognizer, FusedTemporalTransformerRecognizer
from .stabilizer import TemporalSentenceStabilizer
from .text_converter import SignTextConverter

logger = logging.getLogger(__name__)


class _GeminiWorker:
    """Background worker to keep pipeline responsive while calling Gemini."""

    def __init__(self, client: GeminiNativeAudioClient) -> None:
        self.client = client
        self._in_q: "queue.Queue[str]" = queue.Queue(maxsize=1)
        self._out_q: "queue.Queue[dict]" = queue.Queue(maxsize=5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def submit(self, text: str) -> None:
        if not text:
            return
        # Keep only latest request to minimize lag.
        while not self._in_q.empty():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                break
        self._in_q.put_nowait(text)

    def poll_latest(self) -> Optional[dict]:
        latest = None
        while True:
            try:
                latest = self._out_q.get_nowait()
            except queue.Empty:
                return latest

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = asyncio.run(self.client.generate(text))
                payload = {
                    "recognized_text": text,
                    "gemini_text": result.text,
                    "gemini_audio": result.audio_bytes,
                    "gemini_audio_mime": result.audio_mime_type,
                }
                if self._out_q.full():
                    try:
                        self._out_q.get_nowait()
                    except queue.Empty:
                        pass
                self._out_q.put_nowait(payload)
            except Exception as exc:  # pragma: no cover - runtime integration guard
                logger.exception("Gemini worker error: %s", exc)


def run_sign_pipeline_real_time(
    camera_index: int = 0,
    window_size: int = 12,
    max_frames: Optional[int] = None,
    show_preview: bool = False,
    min_send_interval_sec: float = 0.4,
    recognizer: Optional[BaseSignRecognizer] = None,
) -> Generator[Dict[str, object], None, None]:
    """Run real-time sign pipeline and yield incremental structured outputs.

    Yields dict with keys:
    - recognized_text
    - gemini_text
    - gemini_audio
    - gemini_audio_mime
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    recognizer = recognizer or FusedTemporalTransformerRecognizer(window_size=window_size)
    converter = SignTextConverter()
    extractor = MultiStreamLandmarkExtractor(enable_face=False)
    gemini = GeminiNativeAudioClient()
    worker = _GeminiWorker(gemini)

    window: Deque[LandmarkFrame] = deque(maxlen=window_size)
    last_text = ""
    last_sent_at = 0.0

    latest_gemini_text = ""
    latest_gemini_audio = b""
    latest_gemini_mime = "audio/pcm;rate=24000"
    stabilizer = TemporalSentenceStabilizer(converter=converter)

    worker.start()

    try:
        with CameraStream(camera_index=camera_index) as cam:
            for frame_idx, frame in enumerate(cam.frames(), start=1):
                landmark_frame = extractor.extract(frame)
                window.append(landmark_frame)

                tokens = recognizer.predict_tokens(tuple(window))
                stabilized = stabilizer.update(tokens, landmark_frame)
                recognized_text = stabilized.settled_text or stabilized.live_text

                now = time.monotonic()
                if (
                    stabilized.settled_text
                    and stabilized.settled_text != last_text
                    and (now - last_sent_at) >= min_send_interval_sec
                ):
                    worker.submit(stabilized.settled_text)
                    last_text = stabilized.settled_text
                    last_sent_at = now

                gemini_payload = worker.poll_latest()
                if gemini_payload is not None:
                    latest_gemini_text = str(gemini_payload.get("gemini_text", ""))
                    latest_gemini_audio = bytes(gemini_payload.get("gemini_audio", b""))
                    latest_gemini_mime = str(gemini_payload.get("gemini_audio_mime", latest_gemini_mime))

                output = {
                    "recognized_text": recognized_text,
                    "live_text": stabilized.live_text,
                    "settled_text": stabilized.settled_text,
                    "gemini_text": latest_gemini_text,
                    "gemini_audio": latest_gemini_audio,
                    "gemini_audio_mime": latest_gemini_mime,
                }
                yield output

                if show_preview:
                    cv2.imshow("Sign Pipeline Preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if max_frames is not None and frame_idx >= max_frames:
                    break
    finally:
        worker.stop()
        extractor.close()
        if show_preview:
            cv2.destroyAllWindows()
