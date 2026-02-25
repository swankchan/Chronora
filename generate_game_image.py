#!/usr/bin/env python3
"""
Generate a single image from a text prompt (CLI, no Gradio).
Uses SDXL Lightning for speed. Run with: conda activate OpenArch && python generate_game_image.py "your prompt" --output path.png

Usage:
  python generate_game_image.py "dark fantasy dungeon title" --output /path/to/menu_bg.png
  python generate_game_image.py "prompt" --output out.png [--width 1024] [--height 768] [--steps 4]
"""
import os
import sys
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser(description="Generate one image with SDXL Lightning (CLI)")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PNG path (default: outputs/game_<timestamp>.png)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=4, help="Lightning steps (1,2,4,8)")
    parser.add_argument("--negative", type=str, default="blurry, low quality, distorted, text, watermark")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_accelerator = torch.cuda.is_available()
    if has_accelerator:
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.float16 if has_accelerator else torch.float32

    print("Loading SDXL Lightning pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if has_accelerator else None,
    )
    lightning_unet_path = hf_hub_download(
        repo_id="ByteDance/SDXL-Lightning",
        filename="sdxl_lightning_4step_unet.safetensors",
    )
    lightning_unet_dict = load_file(lightning_unet_path)
    pipe.unet.load_state_dict(lightning_unet_dict)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    if has_accelerator:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    print("Pipeline ready. Generating...")

    result = pipe(
        args.prompt,
        negative_prompt=args.negative or None,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        height=args.height,
        width=args.width,
    )
    image = result.images[0]

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        from datetime import datetime
        os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
        out_path = os.path.join(
            os.path.dirname(__file__),
            "outputs",
            f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    image.save(out_path)
    print(f"Saved: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
