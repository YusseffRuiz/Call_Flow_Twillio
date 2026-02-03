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
from llama_cpp import Llama
from twilio.rest import Client


# Your existing modules (from your repo)
from AudioTranscription.ASREngine import AsrEngine
from RAG_CORE.generation_module import GenerationModuleLlama
from RAG_CORE.retrieval_module import RetrievalModule
from voices.TTS_engine import Speaker, speaker_wav_path  # uses XTTS + speaker_wav_path
import campaign_data.utils_script as MESSAGES


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
SILENCE_MS_END_TURN = int(os.getenv("SILENCE_MS_END_TURN", "400"))
MAX_TURN_SECONDS = float(os.getenv("MAX_TURN_SECONDS", "10"))

# Twilio Media Streams audio format
TWILIO_SR = 8000
SAMPLE_WIDTH_BYTES = 2  # PCM16

# Frame size: Twilio commonly sends ~20ms frames
FRAME_MS = 20
FRAME_SAMPLES_8K = int(TWILIO_SR * (FRAME_MS / 1000.0))  # 160 samples @ 8kHz
FRAME_BYTES_PCM16 = FRAME_SAMPLES_8K * SAMPLE_WIDTH_BYTES  # 320 bytes PCM16

LLM_MODEL_FILE = "../HF_Agents/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
HF_TOKEN = os.getenv("HF_TOKEN")
DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DG_API_KEY:
    raise Exception("DEEPGRAM_API_KEY not found")

MEDICAL_EXTENDED = "Documents/medical_life_real.xlsx"

VECTOR_MODEL_NAME = 'jinaai/jina-embeddings-v2-base-es'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
gpu_layers = -1 # 20 a 30 funcionan en nuestra GPU NVIDIA RTX4090 8 GB, -1 settea a GPU
config = {'max_new_tokens': 256, 'context_length': 2048, 'temperature': 0.45, "gpu_layers": gpu_layers,
                          "threads": os.cpu_count()}


DEBUG = True


app = FastAPI(title="Twilio MVP Voice Agent (XTTS + faster-whisper)")


client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# Singletons (for MVP)
ASR = AsrEngine(model_size="medium", device="cuda" if torch.cuda.is_available() else "cpu")
TTS_ENGINE = Speaker(engine="DG", dg_api_key=DG_API_KEY, device="cuda" if torch.cuda.is_available() else "cpu")
"""
Augmentation and Generation Portion
"""
retrieval_module = RetrievalModule(database_path=MEDICAL_EXTENDED, hf_token=HF_TOKEN, model_name=VECTOR_MODEL_NAME)
retrieval_module.initialize(load_db=True, path_to_database="kb_faiss_langchain", score_threshold = 0.34, percentile = 0.9)
llm_model = Llama(model_path=LLM_MODEL_FILE,
                         n_ctx=config["context_length"],
                         # The max sequence length to use - note that longer sequence lengths require much more resources
                         n_threads=config["threads"],
                         # The number of CPU threads to use, tailor to your system and the resulting performance
                         n_gpu_layers=gpu_layers,
                         temperature=config["temperature"],
                         n_batch=512,
                         use_mlock=False,
                         use_mmap=True,
                         f16_kv = True,
                         verbose=False
                         )
llm_module = GenerationModuleLlama(llm_model)
llm_module.initialize(initial_prompt=MESSAGES.INIT_PROMPT_LLAMA, retrieval=retrieval_module, debug=DEBUG)


# -----------------------------
# Text Script (MVP)
# -----------------------------
WELCOME_CAMPAING = [MESSAGES.MSG_1, MESSAGES.MSG_2, MESSAGES.MSG_3, MESSAGES.MID_MESSAGE_1, MESSAGES.MID_MESSAGE_2,
                 MESSAGES.MID_MESSAGE_3, MESSAGES.MID_MESSAGE_4, MESSAGES.MID_MESSAGE_5, MESSAGES.LOCATION_MESSAGE,
                 MESSAGES.CLOSE_MESSAGE_1, MESSAGES.CLOSE_MESSAGE_2]

GREETING_MSG = ("¡Hola! Mi nombre es Cora, soy el asistente de Medical Laif para resolver tus dudas y es un placer"
                " atender tu llamada. Dime, ¿tienes alguna duda sobre un servicio o localización de algún centro?")


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
    min_speech_frames: int = 2
    rms_speech_threshold: int = 500

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
            # PRINT DE CONTROL:
            if self._speech_frames % 10 == 0 and DEBUG:
                print(f"[BUFFERING] Voz acumulada: {self._speech_frames * 20}ms")
        else:
            self._silent_frames += 1
            if self._heard_speech and self._silent_frames % 5 == 0 and DEBUG:
                print(f"[SENSOR] Silencio detectado: {self._silent_frames}/{self._silence_limit_frames}")

        # End-of-turn SOLO si antes hubo speech suficiente
        if self._heard_speech and (self._speech_frames >= self.min_speech_frames):
            if self._silent_frames >= self._silence_limit_frames:
                if DEBUG:
                    print(f"[SENSOR] DISPARO: Turno completo con {len(self._frames)} frames")
                audio = b"".join(self._frames)
                self.reset()
                return True, audio

            # 3. Seguridad: Si el buffer es demasiado grande (ej. 10 segundos), fuerza la transcripción
        if len(self._frames) > self._max_frames:  # 500 * 20ms = 10 segundos
            if DEBUG:
                print("[SENSOR] WARNING: Buffer excedido, forzando transcripción")
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
    Se genera la funcion para el run en paralelo
    """
    return TTS_ENGINE.tts_to_wav(text, language)



# -----------------------------
# Agent "LLM" stub (swap later)
# -----------------------------
def agent_reply(user_text: str, turn_index: int) -> Tuple[str, bool]:
    """
    MVP reply logic.
    """
    pregunta = (user_text or "").strip()
    respuesta, finish_flag = llm_module.rag_answer_stream(query=pregunta)
    # Esperar nueva pregunta
    if "adios" in respuesta.lower() or finish_flag:
        return "Ha sido un gusto ayudarte. ¡Que tengas buen día! ¡Adiós!", True
    if turn_index <= 4:
        return respuesta, finish_flag
    if turn_index == 5: # Ultimo turno
        return f"Agradezco tu tiempo, necesito terminar la llamada, cualquier otra duda, favor de contactar a nuestro número de ayuda.", True
    else:
        return "No logré entender lo que me dijiste, ¿puedes repetirlo?.", False


# -----------------------------
# Twilio HTTP webhook (Answer URL)
# -----------------------------
@app.api_route("/twilio/answer", methods=["GET", "POST"])
async def twilio_answer(request: Request):
    host = request.headers.get("host")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)  # ngrok suele ser https
    ws_scheme = "wss" if proto == "https" else "ws"
    ws_url = f"{ws_scheme}://{host}/twilio/ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="es-MX">Hola. Estoy conectando con el agente.</Say>
    <Connect>
        <Stream url="{ws_url}"/>
    </Connect>
</Response>"""
    if DEBUG:
        print("[ANSWER] method:", request.method)
        print("[ANSWER] headers host/proto:", request.headers.get("host"), request.headers.get("x-forwarded-proto"))
        print("[ANSWER] ws_url:", ws_url)
        print("[ANSWER] twiml:\n", twiml)
    return Response(content=twiml, media_type="application/xml")

# -----------------------------
# Human Transfer
# -----------------------------
async def transfer_to_human(call_sid: str, target_number: str = "+52XXXXXXXXXX"):
    """
    Redirecciona la llamada activa de Twilio a un número de call center,
    sacándola del Media Stream actual.
    """
    try:
        # El TwiML de respuesta saca a la llamada del WebSocket y la rutea
        # Puedes usar <Dial> para un número específico
        client.calls(call_sid).update(
            twiml=f'<Response><Say language="es-MX">Un momento, le transfiero con un asesor humano.</Say><Dial>{target_number}</Dial></Response>'
        )
        if DEBUG:
            print(f"[HANDOFF] Llamada {call_sid} transferida a {target_number}")
    except Exception as e:
        print(f"[ERROR] Handoff failed: {e}")

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
    call_sid: Optional[str] = None

    # State
    detector = TurnDetector(silence_ms=SILENCE_MS_END_TURN)
    bot_task: Optional[asyncio.Task] = None
    bot_speaking = False
    bot_pending = False
    last_bot_end_ts = 0.0
    bot_started_streaming = False
    turn_idx = 0
    barge_speech_hits = 0
    BARGE_MIN_HITS = 3  # 3*20ms = 60ms de voz consistente
    last_tts_task = None
    processing_turn = False

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
            if DEBUG:
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
            if DEBUG:
                print("[DEBUG] bot speaking: ", bot_speaking)
            last_bot_end_ts = time.perf_counter()

    # Funcion usada en vez de speak_text
    async def stream_text_to_speech(sentence: str):
        """Sintetiza y envía una sola oración de forma atómica."""
        try:
            # XTTS Synthesis (Inferencia)
            wav, sr = await asyncio.to_thread(xtts_tts_f32, sentence, LANGUAGE)

            # Resample y Conversión a mu-law
            wav_8k = resample_f32(wav, sr_in=sr, sr_out=TWILIO_SR)
            pcm16 = f32_to_pcm16_bytes(wav_8k)
            mulaw = audioop.lin2ulaw(pcm16, SAMPLE_WIDTH_BYTES)

            # Envío a Twilio
            await send_mulaw_audio(mulaw)
        except asyncio.CancelledError:
            # Manejo de Barge-in: si el usuario interrumpe, se detiene la síntesis
            raise

    # Migrando, a procesamiento paralelo, no usar de momento. Guardando para comparativa.
    async def speak_text(text: str):
        """Generate XTTS audio and stream to Twilio."""
        wav, sr = xtts_tts_f32(text=text, language=LANGUAGE)
        wav = np.asarray(wav, dtype=np.float32)

        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if DEBUG:
            print("[TTS] sr:", sr, "len:", wav.size, "peak:", peak)

        wav_8k = resample_f32(wav, sr_in=sr, sr_out=TWILIO_SR)
        pcm16 = f32_to_pcm16_bytes(wav_8k)
        mulaw = audioop.lin2ulaw(pcm16, SAMPLE_WIDTH_BYTES)
        dur_in = len(wav) / sr
        dur_out = len(mulaw) / TWILIO_SR
        if DEBUG:
            print("[TTS] dur_in", dur_in, "dur_out", dur_out)
        await send_mulaw_audio(mulaw)

    async def start_speaking(text: str, is_partial=False):
        nonlocal bot_task, last_tts_task, bot_pending, bot_started_streaming
        bot_pending = True
        bot_started_streaming = False

        # 1. Manejo de Interrupción (Barge-in)
        if not is_partial:
            if bot_task and not bot_task.done():
                bot_task.cancel()
                try:
                    await bot_task
                except asyncio.CancelledError:
                    pass
            last_tts_task = None  # Reset de la cola

        # 2. Definición de la tarea de habla
        # Capturamos la referencia actual de la cola ANTES de crear la nueva tarea
        prev_task = last_tts_task

        async def speech_wrapper():
            nonlocal bot_pending
            try:
                # Si hay una tarea previa en la cola, esperamos a que termine
                if prev_task and not prev_task.done():
                    await prev_task

                # Ejecutamos la síntesis y envío actual
                await stream_text_to_speech(text)
            except asyncio.CancelledError:
                # Si se cancela durante la espera o el proceso, propagamos
                raise
            finally:
                # Solo si esta es la última tarea de la cola, marcamos como no pendiente
                if last_tts_task == asyncio.current_task():
                    bot_pending = False

        # 3. Creación y asignación de la tarea
        new_task = asyncio.create_task(speech_wrapper())
        bot_task = new_task  # Para el control global de cancelación
        last_tts_task = new_task  # Para el encadenamiento del stream

    # Funcion guardada para comparativa, no usar de momento.
    async def handle_user_turn(pcm16_turn: bytes, call_sid: str):
        """ASR -> agent reply -> next prompt / goodbye."""
        nonlocal turn_idx

        # Convert 8k PCM16 bytes -> float32 -> 16k for ASR
        wav_8k = pcm16_bytes_to_f32(pcm16_turn)
        peak = float(np.max(np.abs(wav_8k))) if wav_8k.size else 0.0
        if peak >= 0.2:
            wav_8k = (wav_8k / peak) * 0.9
        wav_16k = resample_f32(wav_8k, sr_in=TWILIO_SR, sr_out=16000)
        if DEBUG:
            print("[ASR] peak:", peak, "secs:", len(wav_8k) / TWILIO_SR)
        # ASR (faster-whisper)
        # tmp_path = save_wav_tmp_float32_16k(wav_16k, sr=16000)
        segments, _ = ASR.model.transcribe(wav_16k, language=LANGUAGE, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments)

        if not text:
            await start_speaking("Perdón, no te entendí. ¿Puedes repetirlo?")
            return
        if DEBUG:
            print("[ASR] text:", text)

        # Consumo del Strem
        reply = agent_reply(text, turn_idx)  # [0] = real reply, [1] = finish flag

        if "[TRANSFER_CALL]" in reply[0]:
            # 1. Obtener el Call SID
            if call_sid:
                await start_speaking("Entendido, lo comunico con un asesor humano en este momento.")
                # Esperamos un poco para que el TTS se envíe o lo ejecutamos post-transfer
                await transfer_to_human(call_sid)
                await ws.close()
                return

        if reply[1]: ## Closing message
            # Wait for TTS task to finish sending before closing
            await start_speaking(reply[0])
            if bot_task:
                try:
                    await bot_task
                except asyncio.CancelledError:
                    pass
            await ws.close()
            return
        # Then next question / goodbye per turn index
        else:
            if turn_idx <= 4:
                await start_speaking(reply[0])
                turn_idx += 1
            elif turn_idx == 5:
                await start_speaking(reply[0])
                # Wait for TTS task to finish sending before closing
                if bot_task:
                    try:
                        await bot_task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                return

    async def handle_user_turn_stream(pcm16_turn: bytes, call_sid: str):
        """ASR -> agent reply -> next prompt / goodbye."""
        nonlocal turn_idx, bot_task

        # --- PARTE 1: ASR EN MEMORIA ---
        wav_8k = pcm16_bytes_to_f32(pcm16_turn)
        peak = float(np.max(np.abs(wav_8k))) if wav_8k.size else 0.0
        if peak >= 0.2:
            wav_8k = (wav_8k / peak) * 0.9
        wav_16k = resample_f32(wav_8k, sr_in=TWILIO_SR, sr_out=16000)
        if DEBUG:
            print("[DEBUG] Audio gathered, starting transcribe")
        # Inferencia de Faster-Whisper
        segments, _ = ASR.model.transcribe(wav_16k, language=LANGUAGE, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments)

        if not text:
            # Aquí is_partial=False porque es una respuesta de error única
            await start_speaking("Perdón, no te entendí. ¿Puedes repetirlo?", is_partial=False)
            return
        if DEBUG:
            print(f"[ASR] User said: {text}")

        # --- PARTE 2: CONSUMO DEL LLM STREAM ---
        full_response_text = ""
        end_flag = False
        # Creamos un bloque try para manejar la interrupción total si fuera necesario
        async for response in llm_module.rag_answer_stream(text):
            sentence = response['text']
            if not sentence:
                continue

            if response['end_session']:
                print("[DEBUG] Terminando sesion")
                full_response_text += " Entendido, muchas gracias por tu llamada, adios."
                return
            elif "TRANSFER_CALL" in sentence:
                if DEBUG:
                    print("[DEBUG] Transfiriendo a un asesor humano")
                if call_sid:
                    await start_speaking("Entendido, lo comunico con un asesor humano.", is_partial=False)
                    await transfer_to_human(call_sid)
                await ws.close()
                return
            else:
                # Hablamos la oración. Usamos is_partial=True para que se encole/procese fluído.
                await start_speaking(sentence, is_partial=True)
                full_response_text += " " + sentence
                if DEBUG:
                    print("[ASR] sentence text:", sentence)

        # --- PARTE 3: GESTIÓN DE ESTADOS Y CIERRE ---
        turn_idx += 1
        if DEBUG:
            print("[DEBUG] Turno en cuestion: ", turn_idx)

        if "adiós" in full_response_text.lower() or turn_idx > 5 or end_flag:
            # Esperamos a que termine de hablar antes de colgar
            if bot_task and not bot_task.done():
                try:
                    await bot_task
                except asyncio.CancelledError:
                    pass
            await asyncio.sleep(1.5)
            if DEBUG:
                print("[WS] Cerrando conexión por fin de turno o despedida.")
            await ws.close()

    async def safe_handle_turn(audio: bytes, sid: str):
        nonlocal processing_turn, turn_idx
        if processing_turn:
            return

        processing_turn = True
        try:
            # Procesamos el turno (ASR -> RAG -> LLM -> TTS)
            await handle_user_turn_stream(audio, sid)
        except Exception as e:
            print(f"[ERROR] Fallo en el flujo del turno: {e}")
            # Opcional: Enviar un mensaje de error vocal al usuario
            await start_speaking("Lo siento, tuve un problema técnico. ¿Podrías repetir eso?", is_partial=False)
        finally:
            processing_turn = False

    try:
        # Start: greeting + question 1
        while True:
            try:
                # Usamos wait_for para que el socket no se quede bloqueado infinitamente
                # Si en 1 segundo no llega nada de Twilio (ej. Mute total), verificamos el estado
                raw = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                msg = json.loads(raw)
                event = msg.get("event")
            except asyncio.TimeoutError:
                # --- WATCHDOG ---
                # Si el detector escuchó algo pero se quedó atorado por el mute
                # (no se procesa, porque el sistema espera recibir cierto nivel de ruido.)
                if detector._heard_speech and not processing_turn and not bot_speaking:
                    if DEBUG:
                        print("[WATCHDOG] Inactividad (Mute) detectada. Forzando disparo...")
                    audio = b"".join(detector._frames)
                    detector.reset()
                    # Usamos create_task para no bloquear el loop del socket
                    asyncio.create_task(safe_handle_turn(audio, call_sid))
                continue

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"]["callSid"]
                # Start scripted greeting + Q1
                if DEBUG:
                    print("TWILIO WS START:", msg["start"])
                    print("[WS] Stream SID:", stream_sid)
                    print("[WS] Call SID:", call_sid)
                await start_speaking(GREETING_MSG, is_partial=False)
                continue

            if event == "media":
                if not stream_sid or processing_turn:
                    # ignore until start arrives
                    continue

                if DEBUG:
                    print("[DEBUG] Starting second turn")

                payload_b64 = msg["media"]["payload"]
                pcm16_frame = ulaw_b64_to_pcm16(payload_b64)

                frame_bytes = pcm16_frame if isinstance(pcm16_frame, (bytes, bytearray)) else pcm16_frame.tobytes()
                if len(frame_bytes) != FRAME_BYTES_PCM16:  # 320 debug
                    if DEBUG:
                        print("[IN] bad frame size pcm16:", len(frame_bytes))
                    detector.reset()
                    continue

                r = rms_pcm16(frame_bytes)

                # Barge-in: si el bot esta hablando e interrumpen, se limpia y se vuelve a escuchar.
                if bot_speaking:
                    try:
                        # gate por RMS para no disparar por ruido
                        if r > 350 and detector.vad.is_speech(frame_bytes, TWILIO_SR):
                            barge_speech_hits += 1
                            if barge_speech_hits >= BARGE_MIN_HITS:
                                await send_clear()
                                await cancel_bot()
                                detector.reset()
                                barge_speech_hits = 0
                            if DEBUG:
                                print("[BARGE-IN] Voz detectada mientras bot habla")
                        else:
                            barge_speech_hits = max(0, barge_speech_hits - 1)
                        continue

                    except Exception:
                        continue

                if r > 100 and DEBUG:  # Solo printea si hay algo de ruido
                    print(f"[SENSOR] RMS: {int(r)} | VAD: {detector.vad.is_speech(frame_bytes, TWILIO_SR)} | Speaking: {bot_speaking}")
                finished, turn_audio = detector.add_frame(frame_bytes)

                if finished and DEBUG:
                    print(f"[VAD] Turno finalizado. Longitud audio: {len(turn_audio) if turn_audio else 0}")

                if finished and turn_audio and not processing_turn:
                    # opcional: descarta turnos ultra cortos (<0.4s)
                    if len(turn_audio) > int(0.4 * TWILIO_SR * 2):
                        if DEBUG:
                            print("[DEBUG] Starting handler")
                        asyncio.create_task(safe_handle_turn(turn_audio, call_sid))
                    if DEBUG:
                        print("[OUT] Turn: ", turn_idx, "finished:", finished)
                continue

            if event == "stop":
                print("TWILIO WS STOP")
                break

    except Exception as e:
        # Logs de error
        print(f"[CRITICAL] Error en loop de WebSocket: {type(e).__name__} - {e}")
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
