# Work Log — Agent Startup Fixes

Project: `agent_w_intruptionhandling`  
Branch baseline: `main`  
Date: 2026-03-15

---

## 2026-03-15 16:20–16:30
- Pulled/opened baseline code from `main` and reviewed `app/main.py`.
- Detected runtime blockers:
  - missing Python dependencies (`python-dotenv`, `google-genai`)
  - env loading mismatch (`env` file existed, `.env` expected)
  - model/API mismatch for Gemini Live bidi endpoint.

## 2026-03-15 16:30–16:40
- Installed dependencies in `.venv`:
  - `python-dotenv`
  - `google-genai`
- Updated `app/main.py` to robustly load environment from multiple candidate paths:
  - project `.env`
  - project `env`
  - app `.env`
  - app `env`
- Set safer logging defaults (`INFO`) to avoid leaking sensitive header values in debug websocket logs.

## 2026-03-15 16:40–16:45
- Switched runtime to user-requested model:
  - `models/gemini-2.5-flash-native-audio-preview-12-2025`
- Configured Live API correctly for native-audio model:
  - API version `v1alpha`
  - `response_modalities=["AUDIO"]`
  - `output_audio_transcription` enabled
- Added terminal response extraction logic from:
  - `server_content.output_transcription`
  - `server_content.output_audio_transcription`
  - `server_content.model_turn.parts[*].text`

## 2026-03-15 16:45–16:50
- Verified terminal run with piped input (`Hello`) and confirmed successful response.

## 2026-03-15 16:50–16:58
- Implemented simple web frontend + backend streaming pipeline.
- Added backend server file: `app/server.py`
  - FastAPI app + websocket endpoint `/ws`
  - Gemini Live integration
  - streams transcription chunks and audio chunks to browser
- Added frontend file: `static/index.html`
  - chat UI
  - websocket client
  - Web Audio API playback for PCM chunks
  - typing/connection states
- Installed/verified web deps:
  - `fastapi`
  - `uvicorn[standard]`
  - `aiofiles`

## 2026-03-15 17:00–17:08
- Investigated turn desync bug where second prompt used previous turn state.
- Root cause: stale completion/event ordering across turns in long-lived session.
- Iterated backend turn handling logic and completion guards.

## 2026-03-15 17:08–17:15
- Fixed frontend audio indicator getting stuck on "Playing…":
  - changed from global mutable audio counters to per-turn audio tracking object.
- Added frontend timeout guard so UI can recover if completion signal is delayed.

## 2026-03-15 17:15–17:25
- Finalized robust approach in backend:
  - one Gemini Live session per prompt (turn isolation)
  - prevents stale packets crossing turns
  - always emits `turn_complete`/error completion to unlock UI
- Verified multi-turn websocket behavior with scripted test:
  - `hello`
  - `what are you?`
  - `tell me a joke`
  - each turn completed with audio/text events.

---

## Files Changed From Baseline
- Modified: `app/main.py`
- Added: `app/server.py`
- Added: `static/index.html`
- Added: `work_log.md`

## Notes
- `env` is present in repo working tree and contains runtime settings.
- Local server target: `http://localhost:8000`
