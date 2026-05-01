import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

class ArtistAgent:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        Initializes the Stable Diffusion pipeline.
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[ArtistAgent] Loading model on {self.device}...")
        
        # Load the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None # Setting to None for faster local performance
        )
        self.pipe.to(self.device)
        print("[ArtistAgent] Model loaded successfully.")

    def generate_avatar(self, animal_description, output_path):
        """
        Generates a stylized avatar based on the vision agent's description.
        """
        # We wrap the description in a "Style Prompt" to ensure an avatar look
        refined_prompt = (
            f"Professional minimalist avatar icon of {animal_description}, "
            "flat design, simple shapes, vibrant colors, white background, "
            "high resolution, vector art style."
        )
        
        negative_prompt = (
            "realistic, 3d render, shadow, complex background, "
            "blurry, distorted, low quality, text, watermark."
        )

        print(f"[ArtistAgent] Generating artwork for: {animal_description}")
        
        image = self.pipe(
            prompt=refined_prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=25 # Balanced for speed/quality
        ).images[0]

        image.save(output_path)
        print(f"[ArtistAgent] Image saved to {output_path}")
