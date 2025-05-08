import os
import io
import requests
import speech_recognition as sr
import pyttsx3
import threading
from pydub import AudioSegment
import openai
import torch
import numpy as np
import soundfile as sf

# ✅ OpenAI API Key
OPENAI_API_KEY = "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"  
openai.api_key = OPENAI_API_KEY

# ✅ Whisper API URL
WHISPER_API_URL = "http://139.84.142.157:9000/asr"

# ✅ Initialize Text-to-Speech (TTS) engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# ✅ Load Silero VAD model
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

device = torch.device("cpu")
vad_model.to(device)

stop_speaking = threading.Event()

def is_speech_detected(audio_bytes):
    """ Uses Silero VAD to detect speech presence. """
    try:
        audio = read_audio(audio_bytes, sampling_rate=8000)
        speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=8000)
        return len(speech_timestamps) > 0
    except AssertionError:
        return False


def detect_speech():
    """ Listens for user input """
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as mic:
        recognizer.adjust_for_ambient_noise(mic)
        print("🎤 Listening for your question...")
        try:
            audio = recognizer.listen(mic, timeout=5, phrase_time_limit=5)
            audio_bytes = io.BytesIO(audio.get_wav_data())
            audio_bytes.seek(0)
            return audio_bytes
        except sr.WaitTimeoutError:
            return None


def process_audio(audio_bytes):
    """ Ensures audio is 16kHz, mono, and WAV format """
    audio = AudioSegment.from_file(audio_bytes, format="wav")
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


def transcribe_with_whisper(audio_bytes):
    """ Transcribes audio using Whisper API """
    processed_audio = process_audio(audio_bytes)
    files = {"audio_file": ("audio.wav", processed_audio, "audio/wav")}
    params = {"task": "transcribe", "language": "en"}
    response = requests.post(WHISPER_API_URL, files=files, params=params)
    return response.text.strip() if response.status_code == 200 else "Error with Whisper API"


def get_openai_response(prompt):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def speak(text):
    """ AI Speech with interruption detection using Silero VAD """
    stop_speaking.clear()
    def run_tts():
        try:
            engine.say(text)
            engine.startLoop(False)
            while engine.isBusy():
                audio = detect_speech()
                if audio and is_speech_detected(audio):
                    stop_speaking.set()
                    engine.stop()
                    print("\n[INFO] AI Speech Interrupted by User")
                    return
                engine.iterate()
            engine.endLoop()
        except RuntimeError:
            pass
    
    tts_thread = threading.Thread(target=run_tts)
    tts_thread.start()
    tts_thread.join()


if __name__ == "__main__":
    try:
        while True:
            print("\nYou can ask a question now.")
            question_audio = detect_speech()
            if question_audio and is_speech_detected(question_audio):
                question_text = transcribe_with_whisper(question_audio)
                print("\nYou:", question_text)
                ai_response = get_openai_response(question_text)
                print("\nAI:", ai_response)
                stop_speaking.clear()
                speak(ai_response)  # ✅ Ensure AI response is spoken every time
    except KeyboardInterrupt:
        print("\n[INFO] Exiting program...")
        os._exit(0)
