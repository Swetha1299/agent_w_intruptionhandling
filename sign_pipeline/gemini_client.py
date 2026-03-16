"""Gemini client wrapper for text + audio responses using native-audio model."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def _load_env() -> None:
    current = Path(__file__).resolve().parent
    project = current.parent
    for candidate in [project / ".env", project / "env", current / ".env", current / "env"]:
        if candidate.exists():
            load_dotenv(candidate)
            return
    load_dotenv()


@dataclass
class GeminiResult:
    """Structured Gemini response."""

    text: str
    audio_bytes: bytes
    audio_mime_type: str


class GeminiNativeAudioClient:
    """Simple per-request Gemini Live client for native-audio responses."""

    def __init__(
        self,
        model: str = "models/gemini-2.5-flash-native-audio-preview-12-2025",
        api_key: Optional[str] = None,
    ) -> None:
        _load_env()
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        self._client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1alpha"})
        self._config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

    async def generate(self, prompt: str, timeout_sec: float = 20.0) -> GeminiResult:
        """Generate text + audio response from prompt."""
        text_chunks: list[str] = []
        audio_chunks: list[bytes] = []
        audio_mime = "audio/pcm;rate=24000"

        async with self._client.aio.live.connect(model=self.model, config=self._config) as session:
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
                turn_complete=True,
            )

            started = asyncio.get_running_loop().time()
            async for response in session.receive():
                if asyncio.get_running_loop().time() - started > timeout_sec:
                    logger.warning("Gemini turn timed out after %.1fs", timeout_sec)
                    break

                server_content = getattr(response, "server_content", None)
                if not server_content:
                    continue

                # Text transcription fields
                for attr_name in ("output_transcription", "output_audio_transcription"):
                    transcription = getattr(server_content, attr_name, None)
                    tx = getattr(transcription, "text", None)
                    if tx:
                        text_chunks.append(tx)

                # Model parts: text and inline audio
                model_turn = getattr(server_content, "model_turn", None)
                for part in getattr(model_turn, "parts", []) or []:
                    if getattr(part, "thought", False):
                        continue

                    part_text = getattr(part, "text", None)
                    if part_text:
                        text_chunks.append(part_text)

                    inline_data = getattr(part, "inline_data", None)
                    if inline_data:
                        raw = getattr(inline_data, "data", None)
                        mime = getattr(inline_data, "mime_type", audio_mime)
                        if raw and "audio" in mime:
                            if isinstance(raw, str):
                                raw = base64.b64decode(raw)
                            audio_chunks.append(raw)
                            audio_mime = mime

                if (
                    getattr(server_content, "turn_complete", False)
                    or getattr(server_content, "generation_complete", False)
                ):
                    break

        merged_text = "".join(text_chunks).strip()
        merged_audio = b"".join(audio_chunks)
        return GeminiResult(text=merged_text, audio_bytes=merged_audio, audio_mime_type=audio_mime)
