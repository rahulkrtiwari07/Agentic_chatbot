'''import asyncio
import websockets
import os

async def send_wav(uri, file_path):
    try:
        async with websockets.connect(uri) as websocket:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            await websocket.send(audio_data)
            print("WAV file sent.")
            with open("received_audio.wav", 'wb') as output_file: # Save received audio to this file.
                async for message in websocket: # Loop to receive streaming data.
                    if isinstance(message, bytes):  # Check if the received message is binary data.
                        output_file.write(message)
                    else:
                        print(f"Received non-binary message: {message}") # handle non-audio messages if necessary
            print("Streaming audio received and saved to received_audio.wav")

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection was closed: {e}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    uri = "ws://localhost:8000/audio" #replace with your websocket url
    file_path = "resonate 1741267606256.wav" #replace with your wav file path
    asyncio.get_event_loop().run_until_complete(send_wav(uri, file_path))'''

def play_audio(audio_bytes):
    """Play received audio using PyAudio."""
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,  # Adjust based on server response
                    output=True)

    stream.write(audio_bytes)
    stream.stop_stream()
    stream.close()
    p.terminate()

import asyncio
import websockets
import pyaudio
import numpy as np
import wave
import io
import speech_recognition as sr

async def stream_audio_wav(uri, sample_rate=8000, block_size=1024, channels=1):
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket. Streaming audio as WAV...")

            recognizer = sr.Recognizer()
            with sr.Microphone(sample_rate=16000) as mic:
                recognizer.adjust_for_ambient_noise(mic)
                print("🎤 Listening for your question...")

                try:
                    audio = recognizer.listen(mic, timeout=20, phrase_time_limit=20)
                    #audio_bytes = audio.get_wav_data()  # Get raw bytes
                    audio_bytes = io.BytesIO(audio.get_wav_data())

                    await websocket.send(audio_bytes)  # Send as bytes
                    print("Audio sent.")

                    audio_response = await websocket.recv()
                    print(f"🔊 Received {len(audio_response)} bytes of audio data.")

                    play_audio(audio_response)
                    
                except sr.WaitTimeoutError:
                    print("Timeout waiting for audio input. Closing connection.")
    
    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket error: {e}")
                

if __name__ == "__main__":
    uri = "ws://localhost:8000/audio"
    asyncio.run(stream_audio_wav(uri))