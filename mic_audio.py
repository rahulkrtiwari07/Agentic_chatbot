import sounddevice as sd
import numpy as np
import wave
import time
from scipy.io.wavfile import write

# ===================== CONFIG =====================
SAMPLE_RATE = 16000          # Hz
CHANNELS = 1
SILENCE_THRESHOLD = 0.01     # Adjust if needed (0.005–0.02 typical)
SILENCE_DURATION = 2.0       # seconds
CHUNK_DURATION = 0.1         # seconds per read
# ==================================================

def record_until_silence():
    print("🎙️ Listening...")

    audio_buffer = []
    silence_start = None

    def is_silent(chunk):
        peak = np.max(np.abs(chunk))
        return peak < SILENCE_THRESHOLD

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32'
    ) as stream:

        while True:
            chunk, _ = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
            chunk = chunk.flatten()
            audio_buffer.append(chunk)

            if is_silent(chunk):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    print("🔇 Silence detected, stopping...")
                    break
            else:
                silence_start = None

    audio_data = np.concatenate(audio_buffer)
    return audio_data


def save_wav(audio, filename="output.wav"):
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write(filename, SAMPLE_RATE, audio_int16)
    print(f"💾 Saved WAV file: {filename}")
    return filename


if __name__ == "__main__":
    audio = record_until_silence()
    wav_file = save_wav(audio)
    print("✅ Done:", wav_file)
