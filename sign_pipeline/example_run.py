"""Example runner for the real-time sign-language assistant pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from sign_pipeline.pipeline import run_sign_pipeline_real_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run pipeline, log outputs, and persist audio responses continuously."""
    out_dir = Path("sign_pipeline_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    last_saved_audio_len = 0
    for item in run_sign_pipeline_real_time(window_size=30, show_preview=True):
        recognized_text = item.get("recognized_text", "")
        gemini_text = item.get("gemini_text", "")
        gemini_audio = item.get("gemini_audio", b"")

        if recognized_text:
            logger.info("Recognized text: %s", recognized_text)

        if gemini_text:
            logger.info("Gemini text: %s", gemini_text)

        # Save only when audio payload changes and non-empty.
        if isinstance(gemini_audio, (bytes, bytearray)) and len(gemini_audio) > 0 and len(gemini_audio) != last_saved_audio_len:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            audio_path = out_dir / f"gemini_audio_{ts}.pcm"
            audio_path.write_bytes(bytes(gemini_audio))
            last_saved_audio_len = len(gemini_audio)
            logger.info("Saved Gemini audio: %s (%d bytes)", audio_path, last_saved_audio_len)


if __name__ == "__main__":
    main()
