import time
from pathlib import Path

import numpy as np
import pyaudio
import requests
import torch
from TTS.utils.radam import RAdam
import torch.serialization
from TTS.api import TTS
import sounddevice as sd
# from kokoro import KPipeline
import soundfile as sf
from scipy.io import wavfile

# model_name = "tts_models/es/mai/tacotron2-DDC"
# model_name = "tts_models/en/ljspeech/glow-tts" # Ingles
clone_model = "tts_models/multilingual/multi-dataset/xtts_v2" # usado para clonacion de voz
# speaker_wav_path = "PruebaVozAdan.wav" # Voz base, cambiar con respecto a cual quieras usar
speaker_wav_path = "voices/Gil.wav" # Voz base, cambiar con respecto a cual quieras usar
model_name = "tts_models/es/css10/vits"
# model_name = "tts_models/multilingual/multi-dataset/your_tts"
kokoro_voice = 'ef_dora'
DG_MODEL = "aura-2-estrella-es"


class Speaker:
    def __init__(self, engine="XTTS", dg_api_key=None, device= "cuda" if torch.cuda.is_available() else "cpu"):
        self.engine = engine
        if engine == "TTS":
            torch.serialization.add_safe_globals([RAdam])
            self.tts = TTS(model_name).to(device)
            self.sr = 22050
        elif engine == "KOKORO":
            pass
            # self.tts = KPipeline(lang_code='e')
            self.sr = 24000
        elif engine == "XTTS":
            self.tts = TTS(clone_model).to(device)
            self.sr = 24000
        elif engine == "DG":
            # self.tts = TTS(clone_model).to(device)
            self.sr = 8000
            self.dg_api_key = dg_api_key
            self.MODEL_NAME = DG_MODEL  #
        else:
            print("Only TTS and KOKORO are supported")

        self.saving_count = 0

    def speak(self, text):
        if self.engine == "TTS":
            wav = self.tts.tts(text)
            sd.play(wav, samplerate=self.sr)
            sd.wait()
        elif self.engine == "KOKORO":
            generator = self.tts(text, voice=kokoro_voice)
            for i, (gs, ps, audio) in enumerate(generator):
                audio = audio.detach().cpu().numpy()
                sd.play(audio, samplerate=self.sr)
                sd.wait()
        elif self.engine == "DG":
            # Configuramos la URL para audio crudo (linear16)
            DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&encoding=linear16&sample_rate={self.sr}"

            headers = {
                "Authorization": f"Token {self.dg_api_key}",
                "Content-Type": "application/json"
            }
            payload = {"text": text}

            # Abrimos el stream de salida de la laptop
            output_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.SAMPLE_RATE,
                output=True
            )

            start_time = time.time()
            first_byte_received = False

            try:
                with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
                    if r.status_code != 200:
                        print(f"❌ Error API: {r.text}")
                        return

                    # Leemos el stream por trozos
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            if not first_byte_received:
                                ttfb = int((time.time() - start_time) * 1000)
                                print(f"⚡ Latencia PyAudio (TTFB): {ttfb}ms")
                                first_byte_received = True

                            # Enviamos el chunk directamente a las bocinas
                            output_stream.write(chunk)

            except Exception as e:
                print(f"💥 Error en reproducción: {e}")
            finally:
                # Esperamos a que termine de sonar y cerramos el stream del turno
                output_stream.stop_stream()
                output_stream.close()
        else:
            raise ValueError("Not Supported Engine")

    def tts_to_wav(self, text, language="es"):
        if self.engine == "TTS":
            wav = self.tts.tts(text=text)
        elif self.engine == "KOKORO":
            wav = self.tts.tts(text=text, voice=kokoro_voice)
        elif self.engine == "XTTS":
            wav = self.tts.tts(text=text, speaker_wav=speaker_wav_path, language=language)
        elif self.engine == "DG":
            # Configuramos la URL para audio crudo (linear16)
            DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&encoding=linear16&sample_rate={self.sr}"

            headers = {
                "Authorization": f"Token {self.dg_api_key}",
                "Content-Type": "application/json"
            }
            payload = {"text": text}

            try:
                response = requests.post(DEEPGRAM_URL, headers=headers, json=payload)
                if response.status_code != 200:
                    print(f"❌ Error API Deepgram: {response.text}")
                    return np.zeros(0), self.sr

                # Leemos los bytes del WAV y los convertimos a un buffer de memoria
                data = np.frombuffer(response.content, dtype=np.int16)
                wav = data.astype(np.float32) / 32768.0
            except Exception as e:
                print(f"💥 Error convirtiendo DG a WAV: {e}")
                return np.zeros(0), self.sr
        else:
            raise ValueError("Not Supported Engine")
        # Forzar numpy float32
        wav = np.asarray(wav, dtype=np.float32)
        return wav, self.sr

    def save_dialog(self, text, path="audio/"):
        Path(path).mkdir(parents=True, exist_ok=True)
        out_path = f"{path}basic_tts_{self.saving_count}.wav"

        if self.engine == "TTS":
            self.tts.tts_to_file(text=text, file_path=out_path)
        elif self.engine == "KOKORO":
            generator = self.tts(text, voice=kokoro_voice)
            for i, (gs, ps, audio) in enumerate(generator):
                audio = audio.detach().cpu().numpy()
                sf.write(out_path, audio, samplerate=self.sr)
        elif self.engine == "XTTS":
            self.tts.tts_to_file(
                text=text, speaker_wav=speaker_wav_path, language="es", file_path=out_path)
        else:
            raise ValueError("Not Supported Engine")
        self.saving_count += 1
        return out_path
    @staticmethod
    def delete_older_audio(path: str = "audio/", ttl_minutes: int = 10):
        now = time.time()
        ttl_seconds = ttl_minutes * 60

        for audio_file in Path(path).glob("*.wav"):
            try:
                file_age = now - audio_file.stat().st_mtime
                if file_age > ttl_seconds:
                    audio_file.unlink()
            except Exception as e:
                print(f"[WARN] Could not delete {audio_file}: {e}")

    @staticmethod
    def delete_audio_immediately(file_path: str):
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
        except Exception as e:
            print(f"[WARN] Could not delete {file_path}: {e}")

    @staticmethod
    def speak_from_path(path):
        data, samplerate = sf.read(path, dtype="float32")
        sd.play(data, samplerate=samplerate)
        sd.wait()

    def close(self):
        """Llamar al cerrar la aplicación"""
        if self.engine == "DG":
            self.p.terminate()
        else:
            print("Not implemented for this engine")
