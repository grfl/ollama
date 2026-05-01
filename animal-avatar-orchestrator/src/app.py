import os
import sys
import uuid
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from core.vision_agent import identify_animal
from core.artist_agent import ArtistAgent
from core.video_agent import VideoAgent

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# --- Model manager -------------------------------------------------------
# RTX 4070 has 12GB VRAM. SD v1.5 (~3GB) + CogVideoX-2b (~8GB) don't fit
# together, so we hot-swap: unload the idle model before loading the other.

_artist: ArtistAgent | None = None
_video: VideoAgent | None = None
_active: str | None = None  # "artist" | "video"


def _free_vram():
    torch.cuda.empty_cache()


def get_artist() -> ArtistAgent:
    global _artist, _video, _active
    if _active != "artist":
        if _video is not None:
            print("[Server] Unloading VideoAgent to free VRAM...")
            del _video
            _video = None
            _free_vram()
        if _artist is None:
            _artist = ArtistAgent()
        _active = "artist"
    return _artist


def get_video() -> VideoAgent:
    global _artist, _video, _active
    if _active != "video":
        if _artist is not None:
            print("[Server] Unloading ArtistAgent to free VRAM...")
            del _artist
            _artist = None
            _free_vram()
        if _video is None:
            _video = VideoAgent()
        _active = "video"
    return _video


# -------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_avatar():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    input_filename = f"{uuid.uuid4().hex}{ext}"
    input_path = os.path.join(INPUT_DIR, input_filename)
    file.save(input_path)

    try:
        animal_features = identify_animal(input_path)
        if "Error" in animal_features or "stopped" in animal_features:
            return jsonify({"error": f"Vision Agent failed: {animal_features}"}), 500

        output_filename = f"avatar_{uuid.uuid4().hex}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        get_artist().generate_avatar(animal_features, output_path)

        return jsonify({
            "description": animal_features,
            "image_url": f"/output/{output_filename}",
        })
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route("/generate-video", methods=["POST"])
def generate_video():
    data = request.get_json()
    if not data or not data.get("prompt", "").strip():
        return jsonify({"error": "A text prompt is required."}), 400

    prompt = data["prompt"].strip()
    output_filename = f"video_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    get_video().generate_video(prompt, output_path)

    return jsonify({"video_url": f"/output/{output_filename}"})


@app.route("/output/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
