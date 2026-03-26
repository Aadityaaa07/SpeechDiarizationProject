import torch
import soundfile as sf
import json
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=token
)

waveform, sample_rate = sf.read("data/processed/13-00426-01.wav")
waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

audio_input = {
    "waveform": waveform,
    "sample_rate": sample_rate
}

diarization = pipeline(audio_input)

speaker_segments = []

for segment, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    speaker_segments.append({
        "start": segment.start,
        "end": segment.end,
        "speaker": speaker
    })

with open("outputs/speakers.json", "w") as f:
    json.dump(speaker_segments, f, indent=2)

print("Diarization saved!")