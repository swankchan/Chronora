#!/usr/bin/env python3
"""
CLI：用 prompts.txt 批次跑 SDXL Base 1.0 純文字生圖。
格式：每行一組，或同一行用 ; 分隔多組。
每組：promptN="...", neg_promptN="...", stepN=50, guidanceN=7.5, numN=50

Usage:
  python run_prompts_txt.py -p prompts.txt
  python run_prompts_txt.py -p prompts.txt --output-dir my_outputs --width 1024 --height 1024
"""
import os
import re
import sys
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message=".*upcast_vae.*", category=FutureWarning)
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionXLPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_base")


def parse_prompts_file(path):
    """Parse prompts.txt: return list of dicts with prompt, neg_prompt, step, guidance, num."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # 先按換行拆，再按 ; 拆，合併成「組」列表
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    groups = []
    for line in raw_lines:
        for part in line.split(";"):
            part = part.strip()
            if not part:
                continue
            groups.append(part)
    parsed = []
    for g in groups:
        prompt_m = re.search(r'prompt\d*="([^"]*)"', g)
        neg_m = re.search(r'neg_prompt\d*="([^"]*)"', g)
        step_m = re.search(r'step\d*=(\d+)', g)
        guidance_m = re.search(r'guidance\d*=([\d.]+)', g)
        num_m = re.search(r'num\d*=(\d+)', g)
        prompt = prompt_m.group(1).strip() if prompt_m else ""
        if not prompt:
            continue
        parsed.append({
            "prompt": prompt,
            "neg_prompt": neg_m.group(1).strip() if neg_m else "",
            "step": int(step_m.group(1)) if step_m else 30,
            "guidance": float(guidance_m.group(1)) if guidance_m else 7.5,
            "num": int(num_m.group(1)) if num_m else 1,
        })
    return parsed


def save_png_with_comment(pil_image, save_path, prompt, neg_prompt, step, guidance):
    comment = f"prompt: {prompt}\nnegative_prompt: {neg_prompt}\nmodel: {MODEL_ID}\nsteps: {step} | guidance_scale: {guidance}"
    pnginfo = PngInfo()
    pnginfo.add_text("Comment", comment)
    pnginfo.add_text("parameters", comment)
    pil_image.convert("RGB").save(save_path, pnginfo=pnginfo)


def main():
    parser = argparse.ArgumentParser(description="Batch text-to-image from prompts.txt (SDXL Base 1.0)")
    parser.add_argument("-p", "--prompts-file", type=str, default="prompts.txt", help="Path to prompts file (default: prompts.txt)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()

    prompts_path = os.path.abspath(args.prompts_file)
    if not os.path.isfile(prompts_path):
        print(f"Error: prompts file not found: {prompts_path}")
        return 1

    groups = parse_prompts_file(prompts_path)
    if not groups:
        print("Error: no valid prompt groups found in file.")
        return 1

    total_images = sum(g["num"] for g in groups)
    print(f"Loaded {len(groups)} prompt group(s), {total_images} images to generate.")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_accelerator = torch.cuda.is_available()
    if has_accelerator:
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.float16 if has_accelerator else torch.float32

    print("Loading SDXL Base 1.0...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if has_accelerator else None,
    )
    if has_accelerator:
        try:
            pipe.enable_model_cpu_offload()
        except RuntimeError:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    print("Pipeline ready.\n")

    width = args.width
    height = args.height
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated = 0

    for group_idx, g in enumerate(groups):
        n = g["num"]
        prompt = g["prompt"]
        neg_prompt = g["neg_prompt"] or None
        steps = g["step"]
        guidance = g["guidance"]
        preview = (prompt[:60] + "...") if len(prompt) > 60 else prompt
        print(f"[Group {group_idx + 1}/{len(groups)}] {preview} | num={n}, steps={steps}, guidance={guidance}")

        for i in range(n):
            if has_accelerator and (generated > 0) and (generated % 5 == 0):
                torch.cuda.empty_cache()
            result = pipe(
                prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )
            img = result.images[0]
            # 檔名：chronora_base_001_001.png = 第1組第1張
            filename = f"chronora_base_{timestamp}_{group_idx + 1:03d}_{i + 1:03d}.png"
            save_path = os.path.join(args.output_dir, filename)
            save_png_with_comment(img, save_path, prompt, g["neg_prompt"], steps, guidance)
            generated += 1
            if n > 1 and (i + 1) % 10 == 0:
                print(f"    {i + 1}/{n} done.")

    print(f"\nDone. Generated {generated} images -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
