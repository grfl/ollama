import io
import ollama
from PIL import Image

def identify_animal(image_path):
    """
    Uses LLaVA to extract key visual attributes from the animal.
    """
    prompt = (
        "Identify the animal in this photo. List its species and "
        "its two main colors. Be concise."
    )

    try:
        # Convert to JPEG in memory — LLaVA crashes on WebP and some other formats
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        response = ollama.generate(model='llava', prompt=prompt, images=[image_bytes])
        return response['response'].strip()
    except Exception as e:
        return f"Error during vision analysis: {str(e)}"
