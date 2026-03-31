def clean_speaker_segments(speaker_segments, min_duration=1.0):

    cleaned = []

    for seg in speaker_segments:
        duration = seg["end"] - seg["start"]

        if duration < min_duration:
            continue

        if cleaned and cleaned[-1]["speaker"] == seg["speaker"]:
            cleaned[-1]["end"] = seg["end"]
        else:
            cleaned.append(seg)

    return cleaned


def merge_close_segments(speaker_segments, gap_threshold=0.5):
    merged = []

    for seg in speaker_segments:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]

        # 🔥 SAME speaker + small gap → merge
        if (
            seg["speaker"] == prev["speaker"]
            and seg["start"] - prev["end"] < gap_threshold
        ):
            prev["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


def rename_speakers(speaker_segments):
    mapping = {}
    count = 1

    for seg in speaker_segments:
        spk = seg["speaker"]

        if spk not in mapping:
            mapping[spk] = f"Person {count}"
            count += 1

        seg["speaker"] = mapping[spk]

    return speaker_segments


def merge_segments(whisper_segments, speaker_segments):

    # ✅ Step 1: clean noise
    speaker_segments = clean_speaker_segments(speaker_segments)

    # 🔥 Step 2: merge fragmented same-speaker segments
    speaker_segments = merge_close_segments(speaker_segments)

    # ✅ Step 3: rename nicely
    speaker_segments = rename_speakers(speaker_segments)

    # 🎯 timeline
    timeline = []
    for s in speaker_segments:
        timeline.append({
            "speaker": s["speaker"],
            "start": round(s["start"], 2),
            "end": round(s["end"], 2)
        })

    # 🧠 transcript
    full_text = " ".join([w["text"] for w in whisper_segments])

    return timeline, full_text

    # src/merge.py (add to your existing merge.py)
import json
from comparison import DiarizationComparison

def merge_segments_with_comparison(whisper_segments, speaker_segments, audio_path):
    """
    Original merge function with added comparison
    """
    # Run your original merge logic here (existing code)
    timeline, transcript = merge_segments(whisper_segments, speaker_segments)
    
    # Add comparison if audio_path is provided
    comparison_report = None
    if audio_path:
        print("\n🔍 Running model comparison...")
        comparator = DiarizationComparison()
        comparison_report = comparator.run_comparison(audio_path)
    
    return timeline, transcript, comparison_report