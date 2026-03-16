import asyncio
import base64
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sign_pipeline.gemini_client import GeminiNativeAudioClient
from sign_pipeline.landmarks import MultiStreamLandmarkExtractor
from sign_pipeline.recognizer import FusedTemporalTransformerRecognizer
from sign_pipeline.stabilizer import TemporalSentenceStabilizer
from sign_pipeline.text_converter import SignTextConverter


# ── Environment ────────────────────────────────────────────────────────────────

def load_environment() -> None:
    app_dir = Path(__file__).resolve().parent
    project_dir = app_dir.parent
    for candidate in [
        project_dir / ".env",
        project_dir / "env",
        app_dir / ".env",
        app_dir / "env",
    ]:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate)
            return
    load_dotenv()


load_environment()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in env file")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Model: %s", MODEL)


# ── Gemini client ──────────────────────────────────────────────────────────────

gemini_client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})

live_config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    output_audio_transcription=types.AudioTranscriptionConfig(),
)


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Frontend client connected")

    try:
        await websocket.send_json({"type": "ready", "model": MODEL})

        # One Live session per prompt keeps turns isolated and prevents stale
        # completion/audio packets from leaking into the next prompt.
        while True:
            data = await websocket.receive_json()
            user_text = data.get("text", "").strip()
            if not user_text:
                continue

            logger.info("User: %s", user_text)

            try:
                async with gemini_client.aio.live.connect(model=MODEL, config=live_config) as session:
                    await session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=user_text)],
                        ),
                        turn_complete=True,
                    )

                    turn_started_at = time.monotonic()
                    saw_turn_content = False

                    async for response in session.receive():
                        server_content = getattr(response, "server_content", None)
                        if not server_content:
                            continue

                        emitted_content = False

                        # Transcription
                        for attr_name in ("output_transcription", "output_audio_transcription"):
                            transcription = getattr(server_content, attr_name, None)
                            text = getattr(transcription, "text", None)
                            if text:
                                await websocket.send_json({"type": "transcription", "text": text})
                                emitted_content = True

                        # Model-turn parts: text + audio
                        model_turn = getattr(server_content, "model_turn", None)
                        for part in getattr(model_turn, "parts", []) or []:
                            if getattr(part, "thought", False):
                                continue

                            part_text = getattr(part, "text", None)
                            if part_text:
                                await websocket.send_json({"type": "transcription", "text": part_text})
                                emitted_content = True

                            inline_data = getattr(part, "inline_data", None)
                            if inline_data:
                                raw = getattr(inline_data, "data", None)
                                mime = getattr(inline_data, "mime_type", "audio/pcm;rate=24000")
                                if raw and "audio" in mime:
                                    if isinstance(raw, bytes):
                                        raw = base64.b64encode(raw).decode()
                                    await websocket.send_json({
                                        "type": "audio",
                                        "data": raw,
                                        "mime": mime,
                                    })
                                    emitted_content = True

                        if emitted_content:
                            saw_turn_content = True

                        if (
                            getattr(server_content, "turn_complete", False)
                            or getattr(server_content, "generation_complete", False)
                        ):
                            elapsed = time.monotonic() - turn_started_at
                            if saw_turn_content or elapsed > 2.0:
                                await websocket.send_json({"type": "turn_complete"})
                                break
            except Exception as turn_exc:
                logger.exception("Turn failed: %s", turn_exc)
                await websocket.send_json({"type": "error", "message": str(turn_exc)})
                await websocket.send_json({"type": "turn_complete"})

    except WebSocketDisconnect:
        logger.info("Frontend client disconnected")
    except Exception as exc:
        logger.exception("Session error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


@app.websocket("/ws-sign")
async def websocket_sign_pipeline(websocket: WebSocket) -> None:
    """Frontend-driven sign pipeline websocket.

    Expects JSON messages:
      {"type": "frame", "frame": "<base64-jpeg>"}
      {"type": "stop"}
    Emits JSON messages:
      {"type": "sign_ready"}
      {"type": "recognized_text", "text": "..."}
      {"type": "gemini_text", "text": "..."}
      {"type": "audio", "data": "<base64-bytes>", "mime": "audio/pcm;rate=24000"}
      {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info("Sign pipeline client connected")

    extractor = MultiStreamLandmarkExtractor(enable_face=False)
    recognizer = FusedTemporalTransformerRecognizer(window_size=12)
    converter = SignTextConverter()
    stabilizer = TemporalSentenceStabilizer(converter=converter)
    sign_gemini = GeminiNativeAudioClient(model=MODEL, api_key=API_KEY)

    window = deque(maxlen=12)
    last_recognized = ""
    last_live = ""
    last_gemini_prompt = ""
    last_gemini_at = 0.0
    gemini_task: asyncio.Task | None = None

    try:
        await websocket.send_json({"type": "sign_ready"})

        while True:
            payload = None
            try:
                payload = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
            except asyncio.TimeoutError:
                payload = None

            if payload:
                msg_type = payload.get("type")
                if msg_type == "stop":
                    break

                if msg_type == "frame":
                    b64_frame = payload.get("frame", "")
                    if b64_frame:
                        try:
                            frame_bytes = base64.b64decode(b64_frame)
                            img_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        except Exception:
                            frame = None

                        if frame is not None:
                            landmark_frame = extractor.extract(frame)
                            window.append(landmark_frame)
                            tokens = recognizer.predict_tokens(tuple(window))
                            stabilized = stabilizer.update(tokens, landmark_frame)
                            recognized_text = stabilized.settled_text
                            live_text = stabilized.live_text

                            if live_text and live_text != last_live:
                                last_live = live_text
                                await websocket.send_json({"type": "live_text", "text": live_text})

                            if recognized_text and recognized_text != last_recognized:
                                last_recognized = recognized_text
                                await websocket.send_json({"type": "recognized_text", "text": recognized_text})

                            now = time.monotonic()
                            if (
                                recognized_text
                                and recognized_text != last_gemini_prompt
                                and (now - last_gemini_at) >= 0.4
                                and gemini_task is None
                            ):
                                last_gemini_prompt = recognized_text
                                last_gemini_at = now
                                gemini_task = asyncio.create_task(sign_gemini.generate(recognized_text, timeout_sec=15.0))

            if gemini_task and gemini_task.done():
                try:
                    result = gemini_task.result()
                    if result.text:
                        await websocket.send_json({"type": "gemini_text", "text": result.text})
                    if result.audio_bytes:
                        await websocket.send_json(
                            {
                                "type": "audio",
                                "data": base64.b64encode(result.audio_bytes).decode(),
                                "mime": result.audio_mime_type,
                            }
                        )
                except Exception as exc:
                    logger.exception("Sign Gemini task failed: %s", exc)
                    await websocket.send_json({"type": "error", "message": str(exc)})
                finally:
                    gemini_task = None

    except WebSocketDisconnect:
        logger.info("Sign pipeline client disconnected")
    except Exception as exc:
        logger.exception("Sign pipeline session error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if gemini_task and not gemini_task.done():
            gemini_task.cancel()
        extractor.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
