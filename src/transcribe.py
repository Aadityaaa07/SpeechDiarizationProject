import warnings
warnings.filterwarnings("ignore")

from faster_whisper import WhisperModel
import json
import sys
import os

# 🔥 get file path from flask
audio_path = sys.argv[1]

# load model once
model = WhisperModel("medium", compute_type="int8")

def transcribe_audio(file_path):

    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        vad_filter=True
    )

    whisper_segments = []

    for segment in segments:
        text = segment.text.strip()

        if len(text) < 3:
            continue

        whisper_segments.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text
        })

    return whisper_segments


# 🔥 RUN SCRIPT
if __name__ == "__main__":

    result = transcribe_audio(audio_path)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/whisper.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ Transcription saved")