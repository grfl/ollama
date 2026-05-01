import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


class VideoAgent:
    def __init__(self, model_id="THUDM/CogVideoX-2b"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[VideoAgent] Loading {model_id} on {self.device}...")

        self.pipe = CogVideoXPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # Offload unused layers to CPU — keeps peak VRAM under 9GB
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        print("[VideoAgent] Model loaded successfully.")

    def generate_video(self, prompt, output_path, num_frames=49, fps=8):
        print(f"[VideoAgent] Generating video for: {prompt}")

        result = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=num_frames,
            guidance_scale=6.0,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        export_to_video(result.frames[0], output_path, fps=fps)
        print(f"[VideoAgent] Video saved to {output_path}")
