import asyncio
import base64
import io
import logging
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


def load_environment() -> None:
    app_dir = Path(__file__).resolve().parent
    project_dir = app_dir.parent
    dotenv_candidates = [
        project_dir / ".env",
        project_dir / "env",
        app_dir / ".env",
        app_dir / "env",
    ]

    for dotenv_path in dotenv_candidates:
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
            return

    load_dotenv()


load_environment()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env or env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("Using MODEL = %s", MODEL)

client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    output_audio_transcription=types.AudioTranscriptionConfig(),
)


def iter_server_text(response) -> list[str]:
    texts: list[str] = []

    server_content = getattr(response, "server_content", None)
    if not server_content:
        return texts

    for attr_name in ("output_transcription", "output_audio_transcription"):
        transcription = getattr(server_content, attr_name, None)
        transcription_text = getattr(transcription, "text", None)
        if transcription_text:
            texts.append(transcription_text)

    model_turn = getattr(server_content, "model_turn", None)
    for part in getattr(model_turn, "parts", []) or []:
        if getattr(part, "thought", False):
            continue
        part_text = getattr(part, "text", None)
        if part_text:
            texts.append(part_text)

    return texts


def pcm_to_wav(pcm_data: bytes, mime_type: str = "audio/pcm;rate=24000") -> bytes:
    """Wrap raw PCM bytes in a WAV container so generate_content can transcribe them."""
    sample_rate = 24000
    for segment in mime_type.split(";"):
        segment = segment.strip()
        if segment.startswith("rate="):
            try:
                sample_rate = int(segment[5:])
            except ValueError:
                pass
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def main():
    logger.info("Connecting to Gemini Live. Press Ctrl+C to exit.\n")

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        logger.info("Connected.\n")

        while True:
            try:
                user_input = await asyncio.get_running_loop().run_in_executor(None, lambda: input("You: "))
            except (EOFError, KeyboardInterrupt):
                logger.info("Exiting.")
                break

            if not user_input.strip():
                continue

            try:
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=user_input)],
                    ),
                    turn_complete=True,
                )
            except Exception as e:
                logger.exception("Error while sending prompt: %s", e)
                continue

            printed_text = False
            audio_chunks: list[bytes] = []
            audio_mime = "audio/pcm;rate=24000"
            interrupted = False
            try:
                print("Gemini: ", end="", flush=True)
                async for response in session.receive():
                    server_content = getattr(response, "server_content", None)
                    
                    # Check for interruption FIRST, before processing anything
                    if server_content and getattr(server_content, "interruption", None):
                        logger.info("Interruption detected. Stopping response.")
                        interrupted = True
                        break

                    for chunk in iter_server_text(response):
                        print(chunk, end="", flush=True)
                        printed_text = True

                    if not server_content:
                        continue

                    # Collect audio bytes for fallback transcription
                    model_turn = getattr(server_content, "model_turn", None)
                    for part in getattr(model_turn, "parts", []) or []:
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data:
                            raw = getattr(inline_data, "data", None)
                            mime = getattr(inline_data, "mime_type", "")
                            if raw and "audio" in mime:
                                if isinstance(raw, str):
                                    raw = base64.b64decode(raw)
                                audio_chunks.append(raw)
                                audio_mime = mime

                    if getattr(server_content, "turn_complete", False) or getattr(server_content, "generation_complete", False):
                        break

                # Fallback: send collected audio to generate_content for transcription
                if not printed_text and audio_chunks and not interrupted:
                    try:
                        wav_data = pcm_to_wav(b"".join(audio_chunks), audio_mime)
                        tr = await client.aio.models.generate_content(
                            model="models/gemini-2.5-flash",
                            contents=[
                                types.Part(inline_data=types.Blob(data=wav_data, mime_type="audio/wav")),
                                types.Part(text="Transcribe this audio verbatim. Output only the spoken words."),
                            ],
                        )
                        transcription = getattr(tr, "text", None)
                        if transcription:
                            print(transcription, end="", flush=True)
                            printed_text = True
                    except Exception as te:
                        logger.warning("Fallback transcription failed: %s", te)

                if not printed_text and not interrupted:
                    print("[No transcription available]", end="", flush=True)
                elif interrupted:
                    print("[Response interrupted]", end="", flush=True)
                print("\n", flush=True)
            except Exception as e:
                logger.exception("Error while receiving response: %s", e)
                print("\n", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
