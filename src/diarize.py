import warnings
warnings.filterwarnings("ignore")

import torch
import soundfile as sf
import json
import os
import sys
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# 🔥 load env
load_dotenv()

file_path = sys.argv[1]

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.getenv("HF_TOKEN")
)

# 🔥 improve clustering
pipeline.instantiate({
    "clustering": {
        "method": "centroid",
        "threshold": 0.7
    }
})

import librosa

def detect_gender(file_path, start, end):
    y, sr = librosa.load(file_path, sr=None)

    # cut segment
    segment_audio = y[int(start*sr):int(end*sr)]

    if len(segment_audio) == 0:
        return "Unknown"

    pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
    pitch = pitches[magnitudes > magnitudes.mean()]

    if len(pitch) == 0:
        return "Unknown"

    avg_pitch = pitch.mean()

    if avg_pitch < 160:
        return "Male"
    else:
        return "Female"

from pydub import AudioSegment

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)

    # convert to mono
    audio = audio.set_channels(1)

    # resample to 16kHz
    audio = audio.set_frame_rate(16000)

    # normalize volume
    audio = audio.apply_gain(-audio.dBFS)

    temp_path = "temp.wav"
    audio.export(temp_path, format="wav")

    return temp_path

def diarize_audio(file_path):

    waveform, sample_rate = sf.read(file_path)

    # 🔥 FIX 1: convert stereo → mono
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    # 🔥 FIX 2: convert to torch tensor
    waveform = torch.from_numpy(waveform).float()

    # 🔥 FIX 3: ensure shape (1, time)
    waveform = waveform.unsqueeze(0)

    audio_input = {
        "waveform": waveform,
        "sample_rate": sample_rate
    }

    diarization = pipeline(audio_input)
    annotation = diarization.speaker_diarization

    speaker_segments = []

    for segment, _, speaker in annotation.itertracks(yield_label=True):
        speaker_segments.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "speaker": speaker
        })

    return speaker_segments


# 🔥 RUN SCRIPT
if __name__ == "__main__":

    result = diarize_audio(file_path)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/speakers.json", "w") as f:
        json.dump(result, f, indent=2)

    print("✅ Diarization saved")


    # src/diarize.py (add to existing code)
import json
import os

def save_diarization_segments(segments, output_path):
    """
    Save diarization segments in a standardized format
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    return output_path

# Add to your existing diarization code after getting segments:
# save_diarization_segments(segments, 'outputs/pyannote_segments.json')