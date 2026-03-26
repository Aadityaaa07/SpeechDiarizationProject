import json

with open("outputs/whisper.json", "r", encoding="utf-8") as f:
    whisper_segments = json.load(f)

with open("outputs/speakers.json", "r") as f:
    speaker_segments = json.load(f)

final_output = []

for w in whisper_segments:
    w_start = w["start"]
    w_end = w["end"]
    text = w["text"]

    speaker_label = "UNKNOWN"

    for s in speaker_segments:
        if w_start >= s["start"] and w_start <= s["end"]:
            speaker_label = s["speaker"]
            break

    line = f"{w_start:.2f} → {w_end:.2f} | {speaker_label} : {text}"
    final_output.append(line)
    print(line)

# Save final output
with open("outputs/final_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(final_output))

print("\nFinal output saved!")