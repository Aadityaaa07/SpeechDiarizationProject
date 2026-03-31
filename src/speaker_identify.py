import torch
import numpy as np
from speechbrain.inference import EncoderClassifier
import soundfile as sf

# load model once
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

def extract_embedding(audio_path, start, end):
    signal, sr = sf.read(audio_path)

    # cut segment
    segment = signal[int(start*sr):int(end*sr)]

    if len(segment) == 0:
        return None

    segment = torch.tensor(segment).unsqueeze(0)

    embedding = classifier.encode_batch(segment)
    return embedding.squeeze().detach().numpy()

    def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def identify_speakers(audio_path, speaker_segments):

    speaker_embeddings = {}
    speaker_map = {}
    person_count = 1

    for seg in speaker_segments:
        emb = extract_embedding(audio_path, seg["start"], seg["end"])

        if emb is None:
            continue

        matched = False

        for known_speaker, known_emb in speaker_embeddings.items():
            sim = cosine_similarity(emb, known_emb)

            if sim > 0.75:   # 🔥 threshold
                speaker_map[seg["speaker"]] = speaker_map[known_speaker]
                matched = True
                break

        if not matched:
            label = f"Person {person_count}"
            speaker_map[seg["speaker"]] = label
            speaker_embeddings[seg["speaker"]] = emb
            person_count += 1

    # apply mapping
    for seg in speaker_segments:
        seg["speaker"] = speaker_map.get(seg["speaker"], seg["speaker"])

    return speaker_segments