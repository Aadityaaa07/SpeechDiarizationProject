from flask import Flask, render_template, request, send_from_directory, session, jsonify
import os
import subprocess
import json
import uuid
from pydub import AudioSegment
from src.merge import merge_segments
from src.comparison import DiarizationComparison
import json

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "data/processed"
TRIM_FOLDER = "data/trimmed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRIM_FOLDER, exist_ok=True)


# 🔥 RUN BOTH ENV PIPELINES
def run_pipeline(file_path):
    print("🔹 Running transcription with Hindi language...")
    subprocess.run(
        ["venv_whisper\\Scripts\\python", "src\\transcribe.py", file_path, "--language", "hi"],
        check=True
    )

    print("🔹 Running diarization...")
    subprocess.run(
        ["venv_diarization\\Scripts\\python", "src\\diarize.py", file_path],
        check=True
    )

    print("✅ Pipeline complete")


@app.route("/audio/<filename>")
def serve_audio(filename):
    # Check both folders for the audio file
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    trim_path = os.path.join(TRIM_FOLDER, filename)
    
    if os.path.exists(upload_path):
        return send_from_directory(UPLOAD_FOLDER, filename)
    elif os.path.exists(trim_path):
        return send_from_directory(TRIM_FOLDER, filename)
    else:
        return "File not found", 404


@app.route("/", methods=["GET", "POST"])
def index():
    timeline = None
    transcript = None
    error = None

    # Check if we have processed results in session
    if "timeline" in session and "transcript" in session:
        timeline = session.get("timeline")
        transcript = session.get("transcript")
        # Clear after use
        session.pop("timeline", None)
        session.pop("transcript", None)

    if request.method == "POST":

        # 📤 Upload
        if "audio" in request.files:
            file = request.files["audio"]

            if file.filename != "":
                # Generate unique filename to avoid conflicts
                original_name = file.filename
                extension = os.path.splitext(original_name)[1]
                unique_filename = f"{uuid.uuid4().hex}_{original_name}"
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(file_path)
                session["file"] = unique_filename
                session["original_name"] = original_name
                # Clear any previous results
                session.pop("timeline", None)
                session.pop("transcript", None)
                print(f"✅ Uploaded: {unique_filename}")

        # 🚀 Process (without trim)
        elif request.form.get("action") == "process" and "file" in session:
            file_name = session["file"]
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            
            # Check if file exists in upload or trimmed folder
            if not os.path.exists(file_path):
                file_path = os.path.join(TRIM_FOLDER, file_name)
                if not os.path.exists(file_path):
                    error = "Audio file not found. Please upload again."
                    return render_template("index.html", 
                                         error=error,
                                         audio_path=None,
                                         file_name=session.get("original_name", "No file"))

            # 🔥 run both envs
            try:
                run_pipeline(file_path)

                # 📂 load outputs
                with open("outputs/whisper.json", encoding='utf-8') as f:
                    whisper_segments = json.load(f)

                with open("outputs/speakers.json", encoding='utf-8') as f:
                    speaker_segments = json.load(f)

                # 🧠 merge
                timeline, transcript = merge_segments(
                    whisper_segments,
                    speaker_segments
                )
                
                # Format Hindi transcript with Devanagari support
                if transcript:
                    transcript = f'<div lang="hi">{transcript}</div>'
                
                print(f"✅ Timeline created with {len(timeline) if timeline else 0} segments")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Pipeline error: {e}")
                error = f"Processing failed: {e}"
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                error = f"Unexpected error: {e}"

    # Get current audio path for display
    audio_path = None
    if "file" in session:
        file_name = session["file"]
        # Check if file exists in trimmed or upload folder
        if os.path.exists(os.path.join(TRIM_FOLDER, file_name)):
            audio_path = f"/audio/{file_name}"
        elif os.path.exists(os.path.join(UPLOAD_FOLDER, file_name)):
            audio_path = f"/audio/{file_name}"

    return render_template(
        "index.html",
        timeline=timeline,
        transcript=transcript,
        audio_path=audio_path,
        file_name=session.get("original_name", session.get("file", "No file")),
        error=error
    )


# ✂️ TRIM API WITH AUTO-PROCESS (NO POPUP)
@app.route("/trim", methods=["POST"])
def trim():
    try:
        data = request.get_json()
        print(f"📥 Trim request received: {data}")

        if "file" not in session:
            return jsonify({"error": "No file in session"}), 400

        current_file = session["file"]
        
        # Check if file exists in upload or trimmed folder
        upload_path = os.path.join(UPLOAD_FOLDER, current_file)
        trim_path = os.path.join(TRIM_FOLDER, current_file)
        
        if os.path.exists(upload_path):
            file_path = upload_path
        elif os.path.exists(trim_path):
            file_path = trim_path
        else:
            return jsonify({"error": "Audio file not found"}), 404

        start_sec = float(data["start"])
        end_sec = float(data["end"])
        
        # Validate trim range
        if start_sec < 0 or end_sec <= start_sec:
            return jsonify({"error": "Invalid trim range"}), 400

        start_ms = start_sec * 1000
        end_ms = end_sec * 1000
        
        print(f"✂️ Trimming from {start_sec}s to {end_sec}s")

        # Load and trim audio
        try:
            audio = AudioSegment.from_file(file_path)
            total_duration_ms = len(audio)
            
            if end_ms > total_duration_ms:
                end_ms = total_duration_ms
                print(f"⚠️ Adjusted end time to {end_ms/1000}s")
            
            trimmed = audio[start_ms:end_ms]
            
            # Save trimmed file with a new name
            original_name = session.get("original_name", "trimmed_audio")
            extension = os.path.splitext(original_name)[1] or ".wav"
            trimmed_filename = f"trimmed_{uuid.uuid4().hex}_{original_name}"
            trimmed_path = os.path.join(TRIM_FOLDER, trimmed_filename)
            
            # Export with proper format
            format_map = {
                '.mp3': 'mp3',
                '.wav': 'wav',
                '.m4a': 'mp4',
                '.flac': 'flac'
            }
            format_type = format_map.get(extension.lower(), 'wav')
            trimmed.export(trimmed_path, format=format_type)
            
            print(f"✅ Trimmed audio saved: {trimmed_path}")
            
            # Update session to point to trimmed file
            session["file"] = trimmed_filename
            
            # Auto-process the trimmed audio with Hindi language
            print("🚀 Auto-processing trimmed audio with Hindi language...")
            try:
                # Run pipeline on trimmed file with Hindi language
                subprocess.run(
                    ["venv_whisper\\Scripts\\python", "src\\transcribe.py", trimmed_path, "--language", "hi"],
                    check=True
                )
                
                subprocess.run(
                    ["venv_diarization\\Scripts\\python", "src\\diarize.py", trimmed_path],
                    check=True
                )
                
                # Load outputs
                with open("outputs/whisper.json", encoding='utf-8') as f:
                    whisper_segments = json.load(f)
                
                with open("outputs/speakers.json", encoding='utf-8') as f:
                    speaker_segments = json.load(f)
                
                # Merge segments
                timeline, transcript = merge_segments(
                    whisper_segments,
                    speaker_segments
                )
                
                # Format Hindi transcript
                if transcript:
                    transcript = f'<div lang="hi">{transcript}</div>'
                
                # Store in session for display after reload
                session["timeline"] = timeline
                session["transcript"] = transcript
                
                print("✅ Auto-processing complete")
                
                # Return success without message (no popup)
                return jsonify({
                    "status": "success",
                    "processed": True,
                    "reload": True  # Signal to reload the page
                })
                
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Auto-processing failed: {e}")
                # Return error but still reload
                return jsonify({
                    "status": "error",
                    "message": "Processing failed. Please try again.",
                    "processed": False,
                    "reload": True
                })
            except Exception as e:
                print(f"⚠️ Auto-processing error: {e}")
                return jsonify({
                    "status": "error",
                    "message": "Processing failed. Please try again.",
                    "processed": False,
                    "reload": True
                })
            
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

    except Exception as e:
        print(f"❌ Trim endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)