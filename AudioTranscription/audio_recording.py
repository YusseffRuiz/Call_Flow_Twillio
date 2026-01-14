import os
import uuid

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad


class AudioRecorder:
    def __init__(self, duration=10, path="_temp_audio", rate = 16000, channels = 1, vad=True, vad_mode=1):
        self.duration = duration
        self.path = path
        self.streaming = False
        self.rate = rate
        self.channels = channels # Mono audio
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if self.duration <= 0:
            self.streaming=True
        if vad:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(vad_mode)  # Sensibilidad: 0-3 (0 = más estricto, 3 = más sensible)
        else:
            self.vad = None

    def ensure_mono_16k(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Convierte a mono y 16 kHz; devuelve float32 [-1,1]."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.rate)
        # a float32 [-1,1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # si viene en int16 rango grande, normaliza
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32767.0
        return audio

    def record_seconds(self, seconds=10):
        if not seconds:
            duration=self.duration
        else:
            duration=seconds
        # print(f"Grabando {duration}s… habla ahora")
        audio = sd.rec(int(duration*self.rate), samplerate=self.rate,
                       channels=self.channels, dtype='int16')
        sd.wait()
        # print("Se acabó la grabación")
        return audio.flatten()

    def save_audio(self, audio, sr=16000, path=None):
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"rec_{uuid.uuid4().hex}.wav")
        audio = self.ensure_mono_16k(audio, sr)  # downsample a 16k mono
        sf.write(str(path), audio, self.rate, subtype="PCM_16")
        print(f"Saved in {str(path)}")
        return path

    def resample(self, audio_array, sr=16000):
        if sr != self.rate:
            input_arr = librosa.resample(audio_array, orig_sr=sr, target_sr=self.rate)
        else:
            input_arr = audio_array
        return input_arr

    def record_until_silence(self, silence_ms=500, max_duration=20):
        if self.vad is None:
            print("[ERROR] No se dio de alta un nivel de vad, usa otra funcion.")
            return []
        print("🟢 Grabando... habla ahora")
        audio_frames = []
        silence_limit = int(silence_ms / 30)
        total_chunks = int(max_duration * 1000 / 30)
        silent_chunks = 0
        chunk_duration = 0.03
        frame_size = int(self.rate * chunk_duration)

        with sd.InputStream(samplerate=self.rate, channels=self.channels, dtype='int16') as stream:
            for _ in range(total_chunks):
                audio_chunk, _ = stream.read(frame_size)
                pcm_chunk = audio_chunk.tobytes()

                if self.vad.is_speech(pcm_chunk, self.rate):
                    audio_frames.append(audio_chunk)
                    silent_chunks = 0
                else:
                    silent_chunks += 1
                    audio_frames.append(audio_chunk)

                if silent_chunks > silence_limit:
                    break

        print("🔴 Grabación finalizada")

        if not audio_frames:
            print("⚠️ No se detectó ningún audio.")
            return

        final_audio = np.concatenate(audio_frames, axis=0)
        return final_audio

