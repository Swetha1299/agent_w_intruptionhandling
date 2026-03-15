import asyncio
import os
import logging

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("MODEL", "models/gemini-2.5")
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

logger.info("Using MODEL = %s", MODEL)

client = genai.Client(http_options={"api_version": "v1alpha"})

config = types.LiveConnectConfig(
    response_modalities=["TEXT"],
)

async def main():
    logger.info("Connecting to Gemini Live...")
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        logger.info("Connected! Type your questions below. Press Ctrl+C to exit.\n")

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: input("You: "))
            except (EOFError, KeyboardInterrupt):
                logger.info("Exiting.")
                break

            if not user_input.strip():
                continue

            try:
                logger.debug("Sending user input (turn): %s", user_input)
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=user_input)]
                    ),
                    turn_complete=True,
                )
            except Exception as e:
                logger.exception("Error while sending client content: %s", e)
                continue

            logger.debug("Awaiting responses from session.receive()")
            try:
                print("Gemini: ", end="", flush=True)
                async for response in session.receive():
                    logger.debug("Raw response object: %r", response)
                    sc = getattr(response, "server_content", None)
                    logger.debug("server_content: %r", sc)
                    if not sc:
                        continue

                    out_tx = getattr(sc, "output_transcription", None)
                    if out_tx and getattr(out_tx, "text", None):
                        logger.debug("output_transcription text: %s", out_tx.text)
                        print(out_tx.text, end="", flush=True)

                    model_turn = getattr(sc, "model_turn", None)
                    if model_turn:
                        for part in getattr(model_turn, "parts", []):
                            logger.debug("model_turn part: %r", part)
                            if getattr(part, "thought", False):
                                continue
                            text = getattr(part, "text", None)
                            if text:
                                print(text, end="", flush=True)

                    if getattr(sc, "turn_complete", False) or getattr(sc, "generation_complete", False):
                        print("\n")
                        logger.debug("Turn complete received from server_content")
                        break
            except Exception as e:
                logger.exception("Error while receiving responses: %s", e)
                print("\n")
                continue

if __name__ == "__main__":
    asyncio.run(main())
