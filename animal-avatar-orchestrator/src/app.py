import os
import sys
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from core.vision_agent import identify_animal
from core.artist_agent import ArtistAgent

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

print("[Server] Loading Stable Diffusion model...")
artist = ArtistAgent()
print("[Server] Ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
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
        artist.generate_avatar(animal_features, output_path)

        return jsonify({
            "description": animal_features,
            "image_url": f"/output/{output_filename}",
        })
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route("/output/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
