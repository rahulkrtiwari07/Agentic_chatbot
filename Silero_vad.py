import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps

# Load audio (replace with your actual path)
wav, sample_rate = torchaudio.load("resonate 1741267606256.wav")

# Convert to mono if stereo
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)

# Resample to 16kHz if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    wav = resampler(wav)

# VAD model expects shape [1, T]
wav = wav.squeeze()  # remove batch dimension if needed

# Load Silero VAD model
model = load_silero_vad()

# Run VAD
speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)

# Print results
for segment in speech_timestamps:
    print(f"Speech from {segment['start']}s to {segment['end']}s")
