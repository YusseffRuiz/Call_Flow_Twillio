import time
from pathlib import Path

import torch
from TTS.utils.radam import RAdam
import torch.serialization
from TTS.api import TTS
import sounddevice as sd


from kokoro import KPipeline
import soundfile as sf


# model_name = "tts_models/es/mai/tacotron2-DDC"
# model_name = "tts_models/en/ljspeech/glow-tts"
clone_model = "tts_models/multilingual/multi-dataset/xtts_v2" # usado para clonacion de voz
# speaker_wav_path = "PruebaVozAdan.wav" # Voz base, cambiar con respecto a cual quieras usar
speaker_wav_path = "voices/Gil.wav" # Voz base, cambiar con respecto a cual quieras usar
model_name = "tts_models/es/css10/vits"
# model_name = "tts_models/multilingual/multi-dataset/your_tts"
kokoro_voice = 'ef_dora'


class Speaker:
    def __init__(self, engine="TTS", device= "cuda" if torch.cuda.is_available() else "cpu"):
        self.engine = engine
        if engine == "TTS":
            torch.serialization.add_safe_globals([RAdam])
            self.tts = TTS(model_name).to(device)
        elif engine == "KOKORO":
            self.tts = KPipeline(lang_code='e')
        elif engine == "XTTS":
            self.tts = TTS(clone_model).to(device)
        else:
            print("Only TTS and KOKORO are supported")

        self.saving_count = 0

    def speak(self, text):
        if self.engine == "TTS":
            wav = self.tts.tts(text)
            sd.play(wav, samplerate=22050)
            sd.wait()
        elif self.engine == "KOKORO":
            generator = self.tts(text, voice=kokoro_voice)
            for i, (gs, ps, audio) in enumerate(generator):
                audio = audio.detach().cpu().numpy()
                sd.play(audio, samplerate=24000)
                sd.wait()
        else:
            raise ValueError("Not Supported Engine")

    def save_dialog(self, text, path="audio/"):
        Path(path).mkdir(parents=True, exist_ok=True)
        out_path = f"{path}basic_tts_{self.saving_count}.wav"

        if self.engine == "TTS":
            self.tts.tts_to_file(text=text, file_path=out_path)
        elif self.engine == "KOKORO":
            generator = self.tts(text, voice=kokoro_voice)
            for i, (gs, ps, audio) in enumerate(generator):
                audio = audio.detach().cpu().numpy()
                sf.write(out_path, audio, samplerate=24000)
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
