import torch
from diffusers import StableDiffusionPipeline
import os

class ArtistAgent:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        Initializes the Stable Diffusion pipeline with VRAM optimizations.
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[ArtistAgent] Loading model on {self.device}...")
        
        # Optimization: Use float16 precision and the fp16 variant to save VRAM
        storage_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=storage_dtype,
                variant="fp16" if self.device == "cuda" else None,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe.to(self.device)

            # Optimization: Enable attention slicing to handle large tensors more efficiently
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                print("[ArtistAgent] Memory optimizations enabled (fp16 + attention slicing).")
            
            print("[ArtistAgent] Model loaded successfully.")
            
        except Exception as e:
            print(f"[ArtistAgent] Error loading model: {e}")
            raise e

    def generate_avatar(self, animal_description, output_path):
        """
        Generates a stylized avatar based on the vision agent's description.
        """
        # Refined prompt to ensure a clean, vector-style avatar
        refined_prompt = (
            f"Professional minimalist avatar icon of {animal_description}, "
            "flat design, simple shapes, vibrant colors, white background, "
            "high resolution, vector art style, centered composition."
        )
        
        negative_prompt = (
            "realistic, photo, 3d render, complex background, "
            "blurry, distorted, low quality, text, watermark, grainy, dark."
        )

        print(f"[ArtistAgent] Generating artwork for: {animal_description}")
        
        # Generator for reproducibility (optional)
        generator = torch.Generator(self.device).manual_seed(42)

        image = self.pipe(
            prompt=refined_prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=30, # Increased slightly for better quality
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        image.save(output_path)
        print(f"[ArtistAgent] Image saved successfully to {output_path}")
