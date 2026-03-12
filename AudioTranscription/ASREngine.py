from typing import Tuple

from faster_whisper import WhisperModel
from deepgram import DeepgramClient

# Patrones clínicos mínimos de ejemplo (puedes ampliarlos luego)
MED_PATTERNS = [
(r"(\bno\b\s+(fiebre|dolor|alergias?))", "negacion"),
(r"(\b\d{2,3}\/\d{2,3}\b)", "presion_arterial"),
(r"(\b\d{2,3}\.?\d?\s?°?C\b)", "temperatura"),
]


class AsrEngine:
    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        """
        model_size: "tiny" | "base" | "small" | "medium" | "large-v3"
        device: "cpu" | "cuda"
        """
        # compute_type "int8_float16" funciona bien en CPU modernas; en GPU puedes usar "float16"
        # self.model = whisper.load_model(model_size, device=device)
        self.model = WhisperModel(model_size, device=device, compute_type="int8_float16")


    def transcribe_file(self, audio: str, language: str = "es", fp16=False, without_timestamps=True
                        ):

        # audio es un array
        # result = self.model.transcribe(
        #     audio,
        #     language=language,
        #     fp16=fp16,
        #     without_timestamps=without_timestamps,
        # )
        segments, info = self.model.transcribe(audio,
            language=language,
            without_timestamps=without_timestamps, vad_filter=False)

        # Materializar el generador
        segments_list = list(segments)
        # Concatenar texto
        text = " ".join(seg.text.strip() for seg in segments_list)
        confidence = float(getattr(info, "language_probability", 0.0))

        # # Confianza aproximada a partir de avg_logprob
        # logps = [seg.avg_logprob for seg in segments_list if seg.avg_logprob is not None]
        #
        # if logps:
        #     avg_logp = sum(logps) / len(logps)
        #     # pasar de log-probabilidad a algo tipo [0,1]
        #     confidence = float(math.exp(avg_logp))
        # else:
        #     confidence = 0.0

        # print(text)
        # print(confidence)
        # text = result.get("text", "")
        # confidence = 0.90 if text else 0.0 # placeholder; faster-whisper no expone prob estable
        return text, confidence, None


class DeepgramAsrEngine:
    def __init__(self, api_key: str, model: str = "nova-2", language: str = "es"):
        self.api_key = api_key
        # Inicializamos el cliente principal
        self.client = DeepgramClient(api_key=api_key)
        self.model = model
        self.language = language

    def transcribe_audio(self, audio_bytes: bytes):
        """
        Transcribe audio recolectado del stream de Twilio.
        """
        try:
            source = {'buffer': audio_bytes}

            options = {
                "model": self.model,
                "language": self.language,
                "smart_format": True,
                "encoding": "linear16",  # Formato nativo de Twilio
                "sample_rate": 8000,  # Frecuencia de telefonía
                "container": "none"  # Audio crudo sin cabecera WAV
            }

            # Llamada directa al SDK v3
            response = self.client.listen.prerecorded.v("1").transcribe_file(source, options)

            # Navegación segura por el JSON de respuesta
            alternatives = response.results.channels[0].alternatives[0]
            text = alternatives.transcript
            confidence = float(alternatives.confidence)

            return text, confidence
        except Exception as e:
            print(f"❌ [Deepgram Error] {e}")
            return "", 0.0