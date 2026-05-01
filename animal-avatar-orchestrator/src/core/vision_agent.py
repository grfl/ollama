import ollama

def identify_animal(image_path):
    """
    Uses LLaVA to extract key visual attributes from the animal.
    """
    prompt = (
        "Identify the animal in this photo. List its species and "
        "its two main colors. Be concise."
    )
    
    try:
        response = ollama.generate(model='llava', prompt=prompt, images=[image_path])
        return response['response'].strip()
    except Exception as e:
        return f"Error during vision analysis: {str(e)}"
