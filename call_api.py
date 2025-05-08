from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import Response
import os
import logging
from faster_whisper import WhisperModel
from twilio.twiml.voice_response import VoiceResponse, Start, Stream
import voice_service as vs
from retrival import Retrieval

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Set up your AI and audio processing components
DEFAULT_MODEL_SIZE = "medium"
es_pass = os.getenv('ELASTICSEARCH_KEY')
retrieval = Retrieval(es_pass=es_pass)
model = WhisperModel(DEFAULT_MODEL_SIZE + ".en", device="cpu", compute_type="int8")

@app.get("/")
async def intro():
    return {"message": "hey"}

@app.api_route("/voice/", methods=["GET", "POST"])
async def voice_endpoint(request: Request):
    """Endpoint to respond to Twilio with TwiML for real-time streaming."""
    response = VoiceResponse()
    start = Start()
    start.stream(url="wss://a8c6-140-82-42-230.ngrok-free.app/ws")  # Update to your WebSocket URL
    response.append(start)
    return response.to_xml()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to receive real-time audio data."""
    await websocket.accept()
    buffer = b""
    logging.info("WebSocket connection established")
    
    while True:
        try:
            data = await websocket.receive_bytes()
            buffer += data  # Append received audio data to buffer
            logging.info("Received audio data from Twilio")

            # Process audio data when buffer is full or at certain intervals
            # This example will not have the exact audio processing steps since Whisper
            # does not support real-time streaming directly without modifications

            transcription = transcribe_audio(buffer)  # Placeholder for real-time transcription
            if transcription:
                print(f"Customer: {transcription}")

                # Generate AI response
                output = retrieval.response_llm(transcription)
                if output:
                    output = output.lstrip()
                    vs.play_text_to_speech(output)
                    print(f"AI Assistant: {output}")

            buffer = b""  # Clear buffer after processing

        except Exception as e:
            print("Error:", e)
            break

    await websocket.close()

def transcribe_audio(buffer):
    """Transcribe buffer data using Whisper model (modify for real-time adaptation)."""
    # Here, Whisper would typically expect a complete audio file. Adjust accordingly.
    segments, info = model.transcribe(buffer, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
