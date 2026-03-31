import os
from pydub import AudioSegment

input_folder = "data/raw"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".mp3"):
        mp3_path = os.path.join(input_folder, file)
        wav_path = os.path.join(output_folder, file.replace(".mp3", ".wav"))
        
        audio = AudioSegment.from_mp3(mp3_path)

        # 🔥 normalize + mono + 16kHz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        audio.export(wav_path, format="wav")
        
        print(f"Converted: {file}")

print("All files converted!")