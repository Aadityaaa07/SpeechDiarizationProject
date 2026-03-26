import whisper
import json

model = whisper.load_model("small")   # 👈 UPGRADE MODEL

result = model.transcribe(
    "data/processed/13-00471-02.wav",
    language="hi",
    task="transcribe",
    fp16=False
)

# DEBUG: print detected language
print("Detected language:", result["language"])

with open("outputs/whisper.json", "w", encoding="utf-8") as f:
    json.dump(result["segments"], f, ensure_ascii=False, indent=2)

print("Transcription saved!")