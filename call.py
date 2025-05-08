from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import voice_service as vs
from retrival import Retrieval

app = FastAPI()

DEFAULT_MODEL_SIZE = "medium"
es_pass = os.getenv('ELASTICSEARCH_KEY')
retrieval = Retrieval(es_pass=es_pass)
retriever = retrieval.retrieve_data()
model = WhisperModel(DEFAULT_MODEL_SIZE + ".en", device="cpu", compute_type="int8")

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(file_path):
    """Record audio and save it to file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    frames = []
    for _ in range(0, int(16000 / 1024 * 10)):  # Record for 10 seconds
        data = stream.read(1024)
        frames.append(data)

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    stream.stop_stream()
    stream.close()
    audio.terminate()

def transcribe_audio(file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

@app.get("/")
async def intro():
    return {"message":"hey"}

@app.post("/voice/")
async def voice_endpoint(file: UploadFile = File(...)):
    """Endpoint to process the audio file and return a response."""
    # Save the uploaded file
    temp_file_path = "temp_audio_chunk.wav"
    
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    # Check if the file is not silent
    samplerate, data = wavfile.read(temp_file_path)
    if is_silence(data):
        os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail="Silent audio file provided.")
    
    # Transcribe audio
    transcription = transcribe_audio(temp_file_path)
    os.remove(temp_file_path)  # Clean up the temporary file

    print(f"Customer: {transcription}")
    
    # Process the transcription and get AI response
    output = retrieval.response_llm(transcription)
    if output:
        output = output.lstrip()
        vs.play_text_to_speech(output)
        print(f"AI Assistant: {output}")
        return {"customer_input": transcription, "ai_response": output}
    
    raise HTTPException(status_code=500, detail="Failed to get AI response.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
