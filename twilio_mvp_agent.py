"""
Twilio Real Voice Agent MVP (Bidirectional Media Streams)

Goal:
- Place ONE outbound call
- Twilio connects bidirectional Media Stream to this server (WSS)
- Agent speaks a greeting (XTTS voice clone)
- Agent asks 2 questions (city/municipality, service)
- Agent listens using your own ASR (faster-whisper via AsrEngine)
- Agent replies briefly (stub "LLM" logic you can swap)
- Agent says goodbye and ends the call

Important:
- This MVP intentionally avoids your full RAG + call-center backlog.
- It validates: WSS streaming, audio codecs, VAD end-of-turn (800ms), barge-in, and TTS->Twilio playback.

How to run (local dev):
1) pip install fastapi uvicorn twilio webrtcvad torchaudio soundfile numpy
2) Ensure your env can load TTS (Coqui) + CUDA if needed.
3) Set env vars (see CONFIG section).
4) uvicorn twilio_mvp_agent:app --host 0.0.0.0 --port 8000

In production you must expose HTTPS + WSS publicly and point Twilio to:
- Answer URL: https://YOUR_DOMAIN/twilio/answer
- Media Stream WS: wss://YOUR_DOMAIN/twilio/ws

"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import audioop
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple, List
import soundfile as sf

import numpy as np
import webrtcvad
import torchaudio
import torch
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response, JSONResponse

# Your existing modules (from your repo)
from AudioTranscription.ASREngine import AsrEngine
from voices.TTS_engine import Speaker, speaker_wav_path  # uses XTTS + speaker_wav_path


# -----------------------------
# CONFIG (env vars)
# -----------------------------
# Public base URL of this server, e.g. "https://voice.yourdomain.com"

load_dotenv("Documents/keys.env")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
if not PUBLIC_BASE_URL:
    # Keep empty in dev; /twilio/answer still works if you call it manually.
    pass

# Twilio credentials (only needed for /twilio/start_call)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")  # Twilio number or verified caller ID
DEFAULT_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "")     # optional for quick testing

# Language
LANGUAGE = os.getenv("AGENT_LANGUAGE", "es")

# Turn-taking / VAD
SILENCE_MS_END_TURN = int(os.getenv("SILENCE_MS_END_TURN", "800"))  # 700-900 recommended; default 800
MAX_TURN_SECONDS = float(os.getenv("MAX_TURN_SECONDS", "20"))

# Twilio Media Streams audio format
TWILIO_SR = 8000
SAMPLE_WIDTH_BYTES = 2  # PCM16

# Frame size: Twilio commonly sends ~20ms frames
FRAME_MS = 20
FRAME_SAMPLES_8K = int(TWILIO_SR * (FRAME_MS / 1000.0))  # 160 samples @ 8kHz
FRAME_BYTES_PCM16 = FRAME_SAMPLES_8K * SAMPLE_WIDTH_BYTES  # 320 bytes PCM16

# TTS output pacing
SEND_REALTIME = True  # if True, pace ~20ms per frame for more natural streaming


app = FastAPI(title="Twilio MVP Voice Agent (XTTS + faster-whisper)")

# Singletons (for MVP)
ASR = AsrEngine(model_size="medium", device="cuda" if torch.cuda.is_available() else "cpu")
TTS_ENGINE = Speaker(engine="XTTS", device="cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Text Script (MVP)
# -----------------------------
GREETING_TEXT = (
    "Hola, buen dia, Mi nombre es Cora y soy un asistente automático. "
    "Esta llamada es solo una prueba rápida."
)

QUESTION_1 = "Para comenzar, dime por favor en que ciudad o municipio te encuentras."
QUESTION_2 = "Gracias. Ahora dime, por favor, qué tipo de servicio estás buscando. Por ejemplo, consulta general o dentista."
GOODBYE_TEXT = (
    "Perfecto. Con esta información es suficiente para la prueba. "
    "Muchas gracias por tu tiempo. Que tengas un excelente día. Hasta luego."
)


# -----------------------------
# Helpers: audio codec conversions for Twilio. Helper code adapted to match
# -----------------------------

def save_wav_tmp_float32_16k(wav_16k: np.ndarray, sr: int = 16000) -> str:
    # Asegura float32 mono
    if wav_16k.ndim > 1:
        wav_16k = wav_16k.squeeze()
    wav_16k = wav_16k.astype(np.float32, copy=False)

    # clamp por seguridad
    wav_16k = np.clip(wav_16k, -1.0, 1.0)

    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, f"tw_{uuid.uuid4().hex}.wav")

    # Guardar como PCM_16, que es lo que tu pipeline ya usa
    sf.write(path, wav_16k, sr, subtype="PCM_16")
    return path

def ulaw_b64_to_pcm16(payload_b64: str) -> bytes:
    """Twilio sends 8kHz mu-law (8-bit) base64. Convert to PCM16 bytes."""
    mulaw = base64.b64decode(payload_b64)
    pcm16 = audioop.ulaw2lin(mulaw, SAMPLE_WIDTH_BYTES)  # bytes PCM16
    return pcm16

def pcm16_to_ulaw_b64(pcm16: bytes) -> str:
    """Convert PCM16 bytes to mu-law base64 for Twilio playback."""
    mulaw = audioop.lin2ulaw(pcm16, SAMPLE_WIDTH_BYTES)
    return base64.b64encode(mulaw).decode("ascii")

def rms_pcm16(pcm16: bytes) -> float:
    import array, math
    a = array.array('h')
    a.frombytes(pcm16)
    if not a:
        return 0.0
    s = 0.0
    for x in a:
        s += float(x) * float(x)
    return math.sqrt(s / len(a))

def resample_f32(wav_f32: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float32 waveform."""
    if sr_in == sr_out:
        return wav_f32.astype(np.float32, copy=False)
    wav = torch.from_numpy(wav_f32.astype(np.float32, copy=False)).unsqueeze(0)
    wav_rs = torchaudio.functional.resample(wav, sr_in, sr_out)
    return wav_rs.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

def pcm16_bytes_to_f32(pcm16: bytes) -> np.ndarray:
    a = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return (a / 32768.0).astype(np.float32, copy=False)

def f32_to_pcm16_bytes(wav_f32: np.ndarray) -> bytes:
    wav = np.clip(wav_f32, -1.0, 1.0)
    i16 = (wav * 32767.0).astype(np.int16)
    return i16.tobytes()


# -----------------------------
# Turn Detector (incremental record-until-silence)
# -----------------------------
@dataclass
class TurnDetector:
    vad_mode: int = 2  # 2 se usa regularmente
    silence_ms: int = SILENCE_MS_END_TURN
    max_turn_seconds: float = MAX_TURN_SECONDS
    sample_rate: int = TWILIO_SR
    frame_ms: int = FRAME_MS
    min_speech_frames: int = 8
    rms_speech_threshold: int = 180

    def __post_init__(self):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.vad_mode)
        self._frames: List[bytes] = []
        self._silent_frames = 0
        self._total_frames = 0

        self._heard_speech = False
        self._speech_frames = 0

        self._silence_limit_frames = max(1, int(self.silence_ms / self.frame_ms))
        self._max_frames = int((self.max_turn_seconds * 1000) / self.frame_ms)

    def reset(self) -> None:
        self._frames.clear()
        self._silent_frames = 0
        self._total_frames = 0
        self._heard_speech = False
        self._speech_frames = 0

    def add_frame(self, pcm16_frame: bytes) -> Tuple[bool, Optional[bytes]]:
        if not pcm16_frame:
            return False, None

        self._total_frames += 1
        self._frames.append(pcm16_frame)

        r = rms_pcm16(pcm16_frame)
        if r < self.rms_speech_threshold:
            is_speech = False
        else:
            try:
                is_speech = self.vad.is_speech(pcm16_frame, self.sample_rate)
            except Exception:
                is_speech = False

        if is_speech:
            self._silent_frames = 0
            self._heard_speech = True
            self._speech_frames += 1
        else:
            self._silent_frames += 1

        # End-of-turn SOLO si antes hubo speech suficiente
        if self._heard_speech and (self._speech_frames >= self.min_speech_frames):
            if self._silent_frames >= self._silence_limit_frames:
                audio = b"".join(self._frames)
                self.reset()
                return True, audio

        # Si llevamos mucho silencio y NO hubo speech suficiente, limpia el buffer
        if self._silent_frames >= self._silence_limit_frames and not (
                self._heard_speech and self._speech_frames >= self.min_speech_frames
        ):
            self.reset()
            return False, None

        return False, None


# -----------------------------
# XTTS wrapper
# -----------------------------
def xtts_tts_f32(text: str, language: str = "es") -> Tuple[np.ndarray, int]:
    """
    Genera el TTS a Wav para enviarlo a reproducir en Twilio (wav_f32, sr).
    """
    # Coqui regresa un 1D numpy array (float32) a in sample rate usualmente de 24000
    wav = TTS_ENGINE.tts.tts(text=text, speaker_wav=speaker_wav_path, language=language)
    # Forzar numpy float32
    wav = np.asarray(wav, dtype=np.float32)
    sr = getattr(getattr(TTS_ENGINE.tts, "synthesizer", None), "output_sample_rate", None)
    if not sr:
        # Coqui TTS generally uses 24000 for XTTS v2
        sr = 22050
    return wav, int(sr)


# -----------------------------
# Agent "LLM" stub (swap later)
# -----------------------------
def agent_reply(user_text: str, turn_index: int) -> str:
    """
    MVP reply logic.
    Replace this with llm_module.rag_answer(query=user_text)[0]
    once the streaming layer is stable.
    """
    user_text = (user_text or "").strip()
    if turn_index == 0:
        return f"Gracias. Entendido. Escuché: {user_text}."
    if turn_index == 1:
        return f"Perfecto. Escuché: {user_text}."
    return "Entendido."


# -----------------------------
# Twilio HTTP webhook (Answer URL)
# -----------------------------
@app.api_route("/twilio/answer", methods=["GET", "POST"])
async def twilio_answer(request: Request):
    print("[ANSWER] method:", request.method)
    print("[ANSWER] headers host/proto:", request.headers.get("host"), request.headers.get("x-forwarded-proto"))
    host = request.headers.get("host")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)  # ngrok suele ser https
    ws_scheme = "wss" if proto == "https" else "ws"
    ws_url = f"{ws_scheme}://{host}/twilio/ws"
    print("[ANSWER] ws_url:", ws_url)

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="es-MX">Hola. Estoy conectando con el agente.</Say>
    <Connect>
        <Stream url="{ws_url}"/>
    </Connect>
</Response>"""
    print("[ANSWER] twiml:\n", twiml)
    return Response(content=twiml, media_type="application/xml")


# -----------------------------
# Optional: start an outbound call (for quick testing)
# -----------------------------
@app.post("/twilio/start_call")
async def twilio_start_call(payload: dict):
    """
    POST JSON:
    {
      "to": "+52...."   (optional; defaults to TWILIO_TO_NUMBER env)
    }
    """
    to_number = payload.get("to") or DEFAULT_TO_NUMBER
    if not to_number:
        return JSONResponse({"error": "Missing 'to' number (payload.to or TWILIO_TO_NUMBER env)."}, status_code=400)

    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and PUBLIC_BASE_URL):
        return JSONResponse(
            {"error": "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, PUBLIC_BASE_URL env vars."},
            status_code=400,
        )
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        call = client.calls.create(
            to=to_number,
            from_=TWILIO_FROM_NUMBER,
            url=f"{PUBLIC_BASE_URL}/twilio/answer",
            status_callback=f"{PUBLIC_BASE_URL}/twilio/status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )
        return {"status": "ok", "call_sid": call.sid}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/twilio/status")
async def twilio_status(request: Request):
    form = await request.form()
    print("[STATUS]", dict(form))
    return {"ok": True}

# -----------------------------
# Twilio Media Streams WebSocket (Bidirectional)
# -----------------------------


@app.websocket("/twilio/ws")
async def twilio_ws(ws: WebSocket):
    await ws.accept()

    stream_sid: Optional[str] = None

    # State
    print("[WS] stream_sid:", stream_sid)
    detector = TurnDetector(silence_ms=SILENCE_MS_END_TURN)
    bot_task: Optional[asyncio.Task] = None
    bot_speaking = False
    bot_pending = False
    bot_started_streaming = False
    turn_idx = 0
    last_bot_end_ts = 0.0
    POST_TTS_IGNORE_MS = 800
    barge_speech_hits = 0
    BARGE_MIN_HITS = 3  # 3*20ms = 60ms de voz consistente

    ws_send_lock = asyncio.Lock()

    async def ws_send(obj: dict):
        async with ws_send_lock:
            await ws.send_text(json.dumps(obj))

    async def send_clear():
        if stream_sid:
            await ws_send({"event": "clear", "streamSid": stream_sid})

    async def cancel_bot():
        nonlocal bot_task, bot_speaking
        if bot_task and not bot_task.done():
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass
        bot_task = None
        bot_speaking = False

    async def send_mulaw_audio(mulaw_bytes: bytes):
        nonlocal bot_speaking, bot_pending, bot_started_streaming, last_bot_end_ts
        bot_speaking = True
        try:
            frame_size = FRAME_SAMPLES_8K  # 160 bytes = 20ms @ 8k
            print("[OUT] sending mulaw bytes:", len(mulaw_bytes), "streamSid:", stream_sid)

            t0 = time.perf_counter()
            sent_frames = 0

            for i in range(0, len(mulaw_bytes), frame_size):
                # si nos cancelan por barge-in, salimos rápido
                if asyncio.current_task().cancelled():
                    raise asyncio.CancelledError()

                chunk = mulaw_bytes[i:i + frame_size]

                if not bot_started_streaming:
                    bot_started_streaming = True
                    bot_pending = False
                if not chunk:
                    continue

                payload_b64 = base64.b64encode(chunk).decode("ascii")
                msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": payload_b64}}
                await ws_send(msg)

                sent_frames += 1
                target = t0 + sent_frames * (FRAME_MS / 1000.0)  # 20ms por frame
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)

        except asyncio.CancelledError:
            # importante: si cancelas por barge-in, no quieres “terminar de mandar”
            raise

        finally:
            bot_speaking = False
            last_bot_end_ts = time.perf_counter()

    async def speak_text(text: str):
        """Generate XTTS audio and stream to Twilio."""
        wav, sr = xtts_tts_f32(text=text, language=LANGUAGE)
        wav = np.asarray(wav, dtype=np.float32)

        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        print("[TTS] sr:", sr, "len:", wav.size, "peak:", peak)

        wav_8k = resample_f32(wav, sr_in=sr, sr_out=TWILIO_SR)
        pcm16 = f32_to_pcm16_bytes(wav_8k)
        mulaw = audioop.lin2ulaw(pcm16, SAMPLE_WIDTH_BYTES)
        dur_in = len(wav) / sr
        dur_out = len(mulaw) / TWILIO_SR
        print("[TTS] dur_in", dur_in, "dur_out", dur_out)
        await send_mulaw_audio(mulaw)

    async def start_speaking(text: str):
        """Cancel any prior bot speech and start a new one."""
        nonlocal bot_task, bot_pending, bot_started_streaming
        bot_pending = True
        bot_started_streaming = False
        if bot_task and not bot_task.done():
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass
        bot_task = asyncio.create_task(speak_text(text))

    async def handle_user_turn(pcm16_turn: bytes):
        """ASR -> agent reply -> next prompt / goodbye."""
        nonlocal turn_idx

        # Convert 8k PCM16 bytes -> float32 -> 16k for ASR
        wav_8k = pcm16_bytes_to_f32(pcm16_turn)
        peak = float(np.max(np.abs(wav_8k))) if wav_8k.size else 0.0
        if peak >= 0.2:
            wav_8k = (wav_8k / peak) * 0.9
        wav_16k = resample_f32(wav_8k, sr_in=TWILIO_SR, sr_out=16000)
        print("[ASR] peak:", peak, "secs:", len(wav_8k) / TWILIO_SR)
        # ASR (faster-whisper)
        tmp_path = save_wav_tmp_float32_16k(wav_16k, sr=16000)
        try:
            text, conf, _ = ASR.transcribe_file(tmp_path, language=LANGUAGE)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        text = (text or "").strip()
        conf = conf or ""
        print("[ASR] text:", text, ", conf:", conf)

        # If nothing recognized, prompt again (keep it short)
        if not text:
            await start_speaking("Perdón, no te entendí. ¿Puedes repetirlo, por favor?")
            return

        # Agent reply (stub)`
        reply = agent_reply(text, turn_idx)
        # Then next question / goodbye per turn index
        if turn_idx == 0:
            await start_speaking(reply + " " + QUESTION_2)
            turn_idx += 1
        elif turn_idx == 1:
            await start_speaking(reply + " " + GOODBYE_TEXT)
            # Wait for TTS task to finish sending before closing
            if bot_task:
                try:
                    await bot_task
                except asyncio.CancelledError:
                    pass
            await ws.close()
            return

    try:
        # Start: greeting + question 1
        # We wait for streamSid first (start event), then greet.
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                # Start scripted greeting + Q1
                print("TWILIO WS START:", msg["start"])
                await start_speaking(GREETING_TEXT + " " + QUESTION_1)
                continue

            if event == "media":
                if not stream_sid:
                    # ignore until start arrives
                    continue
                now = time.perf_counter()

                payload_b64 = msg["media"]["payload"]
                pcm16_frame = ulaw_b64_to_pcm16(payload_b64)

                frame_bytes = pcm16_frame if isinstance(pcm16_frame, (bytes, bytearray)) else pcm16_frame.tobytes()
                if len(frame_bytes) != FRAME_BYTES_PCM16:  # 320 debug
                    print("[IN] bad frame size pcm16:", len(pcm16_frame))
                    detector.reset()
                    continue

                r = rms_pcm16(frame_bytes)
                # print("[IN] rms:", r, "bot_speaking:", bot_speaking, "pending:", bot_pending)

                if (now - last_bot_end_ts) * 1000.0 < POST_TTS_IGNORE_MS:
                    detector.reset()
                    continue

                # Si aún no empezó a streamear el bot (está generando TTS), ignora audio entrante
                if bot_pending and not bot_started_streaming:
                    # detector.reset()
                    continue

                # Barge-in: si el bot esta hablando e interrumpen, se limpia y se vuelve a escuchar.
                if bot_speaking:
                    try:
                        # gate por RMS para no disparar por ruido
                        if r > 350 and detector.vad.is_speech(frame_bytes, TWILIO_SR):
                            barge_speech_hits += 1
                        else:
                            barge_speech_hits = max(0, barge_speech_hits - 1)

                        if barge_speech_hits >= BARGE_MIN_HITS:
                            await send_clear()
                            await cancel_bot()
                            detector.reset()
                            barge_speech_hits = 0
                        else:
                            # si no es speech, ignora mientras bot habla
                            continue
                    except Exception:
                        continue

                finished, turn_audio = detector.add_frame(frame_bytes)

                if finished and turn_audio:
                    # opcional: descarta turnos ultra cortos (<0.4s)
                    if len(turn_audio) < int(0.4 * TWILIO_SR * 2):
                        continue

                    await handle_user_turn(turn_audio)
                    print("[OUT] Turn: ", turn_idx, "finished:", finished)
                continue

            if event == "stop":
                print("TWILIO WS STOP")
                break

    except Exception:
        # Keep logs minimal for MVP; add structured logging later.
        pass
    finally:
        # Cancel bot task if still running
        if bot_task and not bot_task.done():
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass
        try:
            await ws.close()
        except Exception:
            pass
