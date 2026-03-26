import streamlit as st
import whisper
import torch
import soundfile as sf
from pyannote.audio import Pipeline
import json
import os
from dotenv import load_dotenv

load_dotenv()

st.title("🎤 Hindi Speech Diarization")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save file
    with open("data/processed/13-00553-01.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    # ---------------- WHISPER ----------------
    st.write("🔍 Transcribing...")
    model = whisper.load_model("small")

    result = model.transcribe("data/processed/13-00553-01.wav", language="hi", fp16=False)
    whisper_segments = result["segments"]

    # ---------------- DIARIZATION ----------------
    st.write("👥 Detecting speakers...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=os.getenv("")
    )

    waveform, sample_rate = sf.read("data/processed/13-00553-01.wav")
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

    # ---------------- MERGE ----------------
    st.write("🧠 Combining results...")

    final_output = []

    for w in whisper_segments:
        speaker_label = "UNKNOWN"

        for s in speaker_segments:
            if w["start"] >= s["start"] and w["start"] <= s["end"]:
                speaker_label = s["speaker"]
                break

        line = f"{w['start']:.2f}-{w['end']:.2f} | {speaker_label}: {w['text']}"
        final_output.append(line)

    # ---------------- DISPLAY ----------------
    st.subheader("📜 Result")

    for line in final_output:
        st.write(line)