import sounddevice as sd
import numpy as np
import time
import io
import wave
import requests
import os

# ===================== CONFIG =====================
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.1        # seconds
SILENCE_DURATION = 2.0      # seconds
SILENCE_THRESHOLD = 0.002
MIN_SPEECH_PEAK = 0.003

ASR_API_URL = "http://164.52.193.73:5200/english/"
DEBUG_DIR = "debug_audio"  # folder for debug files
os.makedirs(DEBUG_DIR, exist_ok=True)
# =================================================

def save_wav_file(audio: np.ndarray, filename: str):
    """Save numpy audio array as WAV"""
    max_val = np.max(np.abs(audio))
    if max_val < 1e-6:
        print(f"⚠️ Audio too quiet to save: {filename}")
        return
    audio_int16 = np.int16(audio / max_val * 32767)
    path = os.path.join(DEBUG_DIR, filename)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    print(f"💾 Saved WAV: {path}")

def record_until_silence():
    print("🎙️ Listening... Speak now")

    audio_buffer = []
    silence_start = None
    speech_detected = False

    PRE_SPEECH_CHUNKS = int(0.3 / CHUNK_DURATION)  # 300 ms pre-speech
    ring_buffer = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
            chunk = chunk.flatten()

            peak = np.max(np.abs(chunk))

            # Keep rolling pre-speech buffer
            ring_buffer.append(chunk)
            if len(ring_buffer) > PRE_SPEECH_CHUNKS:
                ring_buffer.pop(0)

            if peak > MIN_SPEECH_PEAK:
                if not speech_detected:
                    # 🔥 Prepend pre-speech audio
                    audio_buffer.extend(ring_buffer)
                    save_wav_file(np.concatenate(ring_buffer), "debug_pre_speech.wav")
                speech_detected = True
                silence_start = None
                audio_buffer.append(chunk)
            else:
                if speech_detected:
                    audio_buffer.append(chunk)
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        print("🔇 End of speech")
                        break

    if not speech_detected:
        return None

    audio = np.concatenate(audio_buffer)
    save_wav_file(audio, "debug_full_audio.wav")
    return audio

def audio_to_wav_bytes(audio: np.ndarray) -> io.BytesIO | None:
    max_val = np.max(np.abs(audio))
    if max_val < 1e-6:
        return None
    audio = audio / max_val
    buffer = io.BytesIO()
    audio_int16 = np.int16(audio * 32767)
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    # Debug save
    save_wav_file(audio, "debug_sent_to_asr.wav")
    return buffer

def send_to_asr(audio: np.ndarray) -> str:
    wav_bytes = audio_to_wav_bytes(audio)
    if wav_bytes is None:
        return ""

    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}

    try:
        response = requests.post(ASR_API_URL.strip(), files=files, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("transcription", "")
    except requests.exceptions.RequestException as e:
        print("❌ ASR request failed:", e)
        return ""


def listen_and_transcribe():
    audio = record_until_silence()
    if audio is None:
        return ""
    return send_to_asr(audio)

# ===================== MAIN LOOP =====================
if __name__ == "__main__":
    try:
        while True:
            transcript = listen_and_transcribe()
            if transcript:
                print("🧠 ASR:", transcript)
            else:
                print("⚠️ No speech detected")
    except KeyboardInterrupt:
        print("👋 Exiting")
