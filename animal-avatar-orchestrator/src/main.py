import os
import sys
from core.vision_agent import identify_animal
# from core.artist_agent import create_avatar

def run_workflow(image_name):
    """
    Manages the flow from image analysis to avatar generation.
    """
    input_path = os.path.join("input", image_name)
    output_path = os.path.join("output", f"avatar_{image_name}")

    if not os.path.exists(input_path):
        print(f"[Error] Input file '{image_name}' not found in /input folder.")
        return

    print(f"[*] Starting workflow for: {image_name}")

    # Step 1: Vision Analysis
    print("[*] Calling Vision Agent...")
    animal_data = identify_animal(input_path)
    print(f"[+] Animal features detected: {animal_data}")

    # Step 2: Generation (Placeholder for now)
    print("[*] Initializing Artist Agent...")
    # create_avatar(animal_data, output_path)
    
    print("[!] Workflow finished successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[Usage] python3 src/main.py <image_filename>")
    else:
        run_workflow(sys.argv[1])
