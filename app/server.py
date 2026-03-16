import asyncio
import base64
import logging
import os
import time
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedOK
from google.genai.errors import APIError


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


@app.get("/favicon.ico")
async def favicon():
    return {"message": "no favicon"}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Frontend client connected")

    receiver_task = None
    interrupt_queue = asyncio.Queue()
    event_queue = asyncio.Queue()
    
    async def receive_messages():
        """Continuously receive messages from websocket and put them in queue"""
        try:
            while True:
                data = await websocket.receive_json()
                await interrupt_queue.put(data)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected in receive_messages")
        except ConnectionClosedOK:
            logger.debug("WebSocket closed normally")
        except Exception as e:
            logger.exception("Error receiving message: %s", e)
    
    try:
        await websocket.send_json({"type": "ready", "model": MODEL})
        receiver_task = asyncio.create_task(receive_messages())
        
        # Create ONE persistent session for the entire conversation
        async with gemini_client.aio.live.connect(model=MODEL, config=live_config) as session:
            logger.info("Gemini Live session connected")
            
            async def send_user_messages():
                """Handle incoming user messages"""
                try:
                    while True:
                        try:
                            data = await asyncio.wait_for(interrupt_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        
                        msg_type = data.get("type", "")
                        
                        if msg_type == "interrupt":
                            logger.info("Interrupt signal received")
                            await event_queue.put({"type": "interrupt"})
                            continue
                        
                        user_text = data.get("text", "").strip()
                        if not user_text:
                            continue
                        
                        logger.info("User: %s", user_text)
                        await session.send_client_content(
                            turns=types.Content(
                                role="user",
                                parts=[types.Part(text=user_text)],
                            ),
                            turn_complete=True,
                        )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.exception("Error in send_user_messages: %s", e)
                    await event_queue.put({"type": "error", "message": str(e)})

            async def receive_from_gemini():
                """Receive responses from Gemini and put them in event queue"""
                try:
                    while True:
                        try:
                            async for response in session.receive():
                                server_content = getattr(response, "server_content", None)
                                if not server_content:
                                    continue

                                # Check for interruption flag
                                if getattr(server_content, "interruption", None):
                                    logger.info("Interruption flag received from Gemini")
                                    await event_queue.put({"type": "gemini_interrupted"})
                                    continue

                                # Send transcriptions
                                for attr_name in ("output_transcription", "output_audio_transcription"):
                                    transcription = getattr(server_content, attr_name, None)
                                    text = getattr(transcription, "text", None)
                                    if text:
                                        await event_queue.put({"type": "transcription", "text": text})

                                # Send audio
                                model_turn = getattr(server_content, "model_turn", None)
                                for part in getattr(model_turn, "parts", []) or []:
                                    if getattr(part, "thought", False):
                                        continue

                                    part_text = getattr(part, "text", None)
                                    if part_text:
                                        await event_queue.put({"type": "transcription", "text": part_text})

                                    inline_data = getattr(part, "inline_data", None)
                                    if inline_data:
                                        raw = getattr(inline_data, "data", None)
                                        mime = getattr(inline_data, "mime_type", "audio/pcm;rate=24000")
                                        if raw and "audio" in mime:
                                            if isinstance(raw, bytes):
                                                raw = base64.b64encode(raw).decode()
                                            await event_queue.put({
                                                "type": "audio",
                                                "data": raw,
                                                "mime": mime,
                                            })

                                if getattr(server_content, "turn_complete", False):
                                    await event_queue.put({"type": "turn_complete"})
                                    # The receive() generator ends here, so the loop will naturally exit
                                    # and we'll call receive() again for the next turn
                        except StopAsyncIteration:
                            logger.debug("Receive generator ended, waiting for next turn...")
                            # This is normal - after turn_complete, we need to restart
                            continue
                except asyncio.CancelledError:
                    logger.debug("receive_from_gemini cancelled")
                except (ConnectionClosedOK, APIError) as e:
                    logger.debug("Session ended normally: %s", e)
                except Exception as e:
                    logger.exception("Error in receive_from_gemini: %s", e)
                    await event_queue.put({"type": "error", "message": str(e)})
                finally:
                    await event_queue.put(None)  # Signal end of stream

            # Start concurrent tasks
            send_task = asyncio.create_task(send_user_messages())
            receive_task_gemini = asyncio.create_task(receive_from_gemini())

            try:
                # Process events from Gemini and send to client
                turn_in_progress = False
                while True:
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # Keep listening for user input even if no events from Gemini
                        continue
                    
                    if event is None:
                        logger.info("Gemini session ended")
                        break

                    if event.get("type") == "interrupt":
                        continue  # Don't send back to client, just skip

                    if event.get("type") == "error":
                        try:
                            await websocket.send_json(event)
                        except Exception as e:
                            logger.warning("Failed to send error: %s", e)
                        break

                    if event.get("type") == "gemini_interrupted":
                        try:
                            await websocket.send_json({"type": "interrupted"})
                        except Exception as e:
                            logger.warning("Failed to send interrupted: %s", e)
                        turn_in_progress = False
                        continue

                    try:
                        await websocket.send_json(event)
                    except Exception as e:
                        logger.warning("Failed to send event: %s", e)
                        break
                    
                    # Track turn state
                    if event.get("type") == "turn_complete":
                        logger.debug("Turn complete, ready for next message")
                        turn_in_progress = False
                        continue
                    
                    if event.get("type") in ("transcription", "audio"):
                        turn_in_progress = True

            finally:
                send_task.cancel()
                receive_task_gemini.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
                try:
                    await receive_task_gemini
                except asyncio.CancelledError:
                    pass

    except Exception as exc:
        logger.exception("Session error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception as e:
            logger.warning("Failed to send session error: %s", e)
    finally:
        if receiver_task and not receiver_task.done():
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket connection closed")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
