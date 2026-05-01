import os
import sys
from core.vision_agent import identify_animal
from core.artist_agent import ArtistAgent

def run_workflow(image_name):
    input_path = os.path.join("input", image_name)
    output_filename = f"avatar_{os.path.splitext(image_name)[0]}.png"
    output_path = os.path.join("output", output_filename)

    if not os.path.exists(input_path):
        print(f"[Error] File {input_path} not found.")
        return

    # Phase 1: Vision
    print("[Orchestrator] Starting Vision Phase...")
    animal_features = identify_animal(input_path)
    
    # Check if Vision Agent failed
    if "Error" in animal_features or "stopped" in animal_features:
        print(f"[Critical Error] Vision Agent failed: {animal_features}")
        print("Tip: Try restarting Ollama or check if another process is using the GPU.")
        return 

    print(f"[Orchestrator] Features extracted: {animal_features}")

    # Phase 2: Art Generation
    print("[Orchestrator] Starting Artist Phase...")
    try:
        artist = ArtistAgent()
        artist.generate_avatar(animal_features, output_path)
        print(f"[!] Workflow Complete. Check your avatar in: {output_path}")
    except Exception as e:
        print(f"[Critical Error] Artist Agent failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[Usage] python3 src/main.py <image_name>")
    else:
        run_workflow(sys.argv[1])
