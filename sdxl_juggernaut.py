"""
Chronora 時光 - Juggernaut XL 版本
RunDiffusion/Juggernaut-XL（SDXL 熱門 fine-tune），步數 20-50、guidance_scale 約 7.5。
"""
import os
import torch
import gradio as gr
from datetime import datetime
import time
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

# 1. GPU 診斷
print("=" * 60)
print("🔍 GPU 診斷信息")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 數量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    記憶體: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ PyTorch 無法檢測到 CUDA")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
has_accelerator = torch.cuda.is_available()

if has_accelerator:
    print(f"🚀 正在啟動 GPU 引擎... (Device: {device})")
else:
    print(f"⚠️ 未檢測到 GPU，將使用 CPU 模式（速度較慢）")

# 2. HEDdetector
try:
    from controlnet_aux import HEDdetector
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    print("✅ HEDdetector 載入成功")
except (ImportError, AttributeError) as e:
    print(f"⚠️ HEDdetector 載入失敗: {e}")
    try:
        from controlnet_aux.hed import HEDdetector
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        print("✅ HEDdetector (替代路徑) 載入成功")
    except Exception as e2:
        print(f"❌ HEDdetector 載入完全失敗: {e2}")
        hed = None

# 3. Juggernaut XL（RunDiffusion）
dtype = torch.float16 if has_accelerator else torch.float32

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=dtype
)

# 只載入一次：主 pipeline 用 Juggernaut XL
print("📦 載入 Juggernaut XL（RunDiffusion）...")
# Juggernaut-XL 無 fp16 variant，唔傳 variant；GPU 下 torch_dtype=float16 會喺記憶體做半精度
pipe_t2i = StableDiffusionXLPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL",
    torch_dtype=dtype
)
print("✅ Juggernaut XL 載入成功！")

if has_accelerator:
    try:
        pipe_t2i.enable_model_cpu_offload()
    except RuntimeError as e:
        print(f"⚠️ CPU offload 失敗，改用直接載入到 {device}: {e}")
        pipe_t2i = pipe_t2i.to(device)
else:
    pipe_t2i = pipe_t2i.to(device)

# ControlNet 管道：共用組件
print("⚡ 建立 ControlNet 管道（共用組件）...")
pipe = StableDiffusionXLControlNetPipeline(
    controlnet=controlnet,
    **pipe_t2i.components
)
if not has_accelerator:
    pipe = pipe.to(device)
print("✅ ControlNet 管道就緒！")

# Img2Img：延遲建立，共用組件
pipe_img2img = None

def get_pipe_img2img():
    global pipe_img2img
    if pipe_img2img is None:
        print("⚡ 建立 Img2Img 管道（共用組件）...")
        pipe_img2img = StableDiffusionXLImg2ImgPipeline(**pipe_t2i.components)
        if not has_accelerator:
            pipe_img2img = pipe_img2img.to(device)
        pipe_img2img.enable_attention_slicing()
        print("✅ Img2Img 管道就緒！")
    return pipe_img2img

# 4. 優化
try:
    pipe.enable_attention_slicing()
    pipe_t2i.enable_attention_slicing()
    print("✅ 記憶體優化已開啟")
except Exception as e:
    print("⚠️ 記憶體優化未能啟動")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_juggernaut")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESOLUTION_CHOICES = [
    ("512 × 512", "512x512"),
    ("512 × 768", "512x768"),
    ("768 × 512", "768x512"),
    ("768 × 1024", "768x1024"),
    ("1024 × 768", "1024x768"),
    ("1024 × 1024", "1024x1024"),
    ("1280 × 1280", "1280x1280"),
    ("1536 × 1024", "1536x1024"),
    ("1024 × 1536", "1024x1536"),
]

def to_pil(img):
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    if isinstance(img, dict) and "composite" in img:
        return img["composite"].convert("RGB")
    return None

def get_input_image(sketch_dict, upload_img):
    if upload_img is not None:
        return to_pil(upload_img)
    if sketch_dict is not None and "composite" in sketch_dict and sketch_dict["composite"] is not None:
        return sketch_dict["composite"].convert("RGB")
    return None

def process_sketch(sketch_dict, upload_img, prompt, negative_prompt, text_only, resolution_choice,
                   num_steps, guidance_scale, controlnet_scale, batch_size):
    if prompt == "":
        return "Please enter a text prompt.", []

    batch_size = int(batch_size) if batch_size else 1
    if batch_size < 1:
        batch_size = 1
    if batch_size > 100:
        return "Batch size must be 1–100.", []

    if has_accelerator:
        torch.cuda.empty_cache()
    w, h = resolution_choice.split("x")
    width, height = int(w), int(h)
    composite_img = None if text_only else get_input_image(sketch_dict, upload_img)
    start_time = time.time()

    gen_kwargs = {
        "num_inference_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "height": height,
        "width": width
    }

    if negative_prompt and negative_prompt.strip():
        gen_kwargs["negative_prompt"] = negative_prompt.strip()

    control_image = None
    if composite_img is not None:
        if hed is None:
            return "HEDdetector not loaded; sketch mode unavailable.", []
        control_resolution = min(1024, max(width, height))
        control_image = hed(composite_img, detect_resolution=control_resolution, image_resolution=control_resolution)
        gen_kwargs.update({
            "image": control_image,
            "controlnet_conditioning_scale": float(controlnet_scale)
        })

    outputs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    print(f"🔄 開始批量生成 {batch_size} 張圖片（Juggernaut XL）...")
    for i in range(batch_size):
        print(f"   生成第 {i+1}/{batch_size} 張...")

        if composite_img is None:
            result = pipe_t2i(prompt, **gen_kwargs)
        else:
            result = pipe(prompt, **gen_kwargs)

        output = result.images[0]
        outputs.append(output)

        if batch_size > 1:
            filename = f"chronora_juggernaut_{timestamp}_{i+1:03d}.png"
        else:
            filename = f"chronora_juggernaut_{timestamp}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        output.save(save_path)
        saved_files.append(filename)

        if has_accelerator and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    end_time = time.time()
    total_ms = (end_time - start_time) * 1000
    total_sec = total_ms / 1000
    avg_ms = total_ms / batch_size
    mode = "Text only" if composite_img is None else "Sketch+text"
    quality_info = f"steps:{int(num_steps)} | guidance:{float(guidance_scale):.1f} | Juggernaut XL"

    if batch_size > 1:
        status_msg = f"Done: {mode} | {batch_size} images | {quality_info} | {total_sec:.1f}s | {avg_ms:.0f} ms/img | Saved: {saved_files[0]} (+{batch_size} files)"
    else:
        status_msg = f"Done: {mode} | {quality_info} | {total_ms:.2f} ms | Saved: {saved_files[0]}"
    return status_msg, outputs

def process_modify(init_image, modify_prompt, negative_prompt, strength, resolution_choice, num_steps, guidance_scale):
    img = to_pil(init_image)
    if img is None:
        return "Please upload an image to modify.", []
    if not modify_prompt or not modify_prompt.strip():
        return "Please enter a modify prompt.", []

    w, h = resolution_choice.split("x")
    width, height = int(w), int(h)
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    start_time = time.time()
    if has_accelerator:
        torch.cuda.empty_cache()
    p = get_pipe_img2img()
    kwargs = {
        "prompt": modify_prompt.strip(),
        "image": img,
        "strength": float(strength),
        "num_inference_steps": int(num_steps),
        "guidance_scale": float(guidance_scale)
    }
    if negative_prompt and negative_prompt.strip():
        kwargs["negative_prompt"] = negative_prompt.strip()
    output = p(**kwargs).images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chronora_juggernaut_mod_{timestamp}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    output.save(save_path)

    ms = (time.time() - start_time) * 1000
    status_msg = f"Img2Img done | steps:{int(num_steps)} | guidance:{float(guidance_scale):.1f} | {ms:.2f} ms | Saved: {filename}"
    return status_msg, [output]

# 5. UI
HEADING_AND_LABEL_CSS = """
.gradio-container { background: #4a4a4a !important; }
.gradio-container .block, .gradio-container .form, .gradio-container .panel { background: #4a4a4a !important; }
#app-heading, #app-heading p, #app-heading * { font-size: 2.2rem !important; font-weight: 600 !important; color: #e0e0e0 !important; }
.gradio-container label { font-size: 0.8rem !important; font-weight: 500 !important; color: #d0d0d0 !important; }
.gradio-container .markdown p { color: #d8d8d8 !important; }
.gradio-container input, .gradio-container textarea { background: #3a3a3a !important; color: #e8e8e8 !important; border: 1px solid #555 !important; outline: none !important; }
.gradio-container select, .gradio-container [data-testid="dropdown"], .gradio-container .gr-box { background: #3a3a3a !important; color: #e8e8e8 !important; border: 1px solid #555 !important; outline: none !important; }
"""

with gr.Blocks(css=HEADING_AND_LABEL_CSS) as demo:
    gr.Markdown("Chronora — Juggernaut XL (RunDiffusion)", elem_id="app-heading")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Sketch"):
                    input_sketch = gr.Sketchpad(label="Draw sketch", type="pil", layers=False)
                with gr.TabItem("Upload"):
                    input_upload = gr.Image(label="Upload sketch or reference", type="pil", sources=["upload", "clipboard"])
            text_only_check = gr.Checkbox(label="Text-only (no sketch)", value=False)
            resolution_dropdown = gr.Dropdown(
                choices=RESOLUTION_CHOICES,
                value="1024x1024",
                label="Resolution"
            )
            with gr.Accordion("Quality", open=False):
                num_steps_slider = gr.Slider(
                    minimum=20,
                    maximum=50,
                    value=30,
                    step=1,
                    label="Steps (20–50, default 30)"
                )
                guidance_scale_slider = gr.Slider(
                    minimum=5.0,
                    maximum=9.0,
                    value=7.5,
                    step=0.1,
                    label="Guidance scale (5–9, default 7.5)"
                )
                controlnet_scale_slider = gr.Slider(
                    minimum=0.3,
                    maximum=1.5,
                    value=0.8,
                    step=0.1,
                    label="ControlNet strength (sketch mode)"
                )
                batch_size_input = gr.Number(
                    value=1,
                    minimum=1,
                    maximum=100,
                    step=1,
                    label="Batch size (1–100)"
                )
                gr.Markdown("Steps 20–30 faster; 30–50 better quality. Output: `outputs_juggernaut/`")
            prompt_txt = gr.Textbox(
                label="Prompt (required)",
                placeholder="e.g. A majestic cat, cinematic lighting, 8k, highly detailed",
                lines=3
            )
            negative_prompt_txt = gr.Textbox(
                label="Negative prompt (optional)",
                placeholder="blurry, deformed, low quality, ugly, bad anatomy",
                lines=1
            )
            gr.Markdown("**Presets**")
            with gr.Row():
                btn_arch = gr.Button("Architecture", size="sm")
                btn_interior = gr.Button("Interior", size="sm")
                btn_cyber = gr.Button("Cyberpunk", size="sm")
                btn_minimal = gr.Button("Minimal", size="sm")
            with gr.Row():
                btn_anime = gr.Button("Anime", size="sm")
                btn_photo = gr.Button("Photo", size="sm")
                btn_watercolor = gr.Button("Watercolor", size="sm")
                btn_fantasy = gr.Button("Fantasy", size="sm")
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            output_gallery = gr.Gallery(
                label="Output",
                show_label=True,
                elem_id="gallery",
                columns=4,
                rows=3,
                height="auto"
            )
            timer_lbl = gr.Label(value="Ready.", label="")
            with gr.Accordion("Img2Img", open=False):
                gr.Markdown("Upload an image and enter a new prompt to vary it.")
                modify_image_input = gr.Image(label="Image to modify", type="pil", sources=["upload", "clipboard"])
                modify_prompt_txt = gr.Textbox(label="Modify prompt", placeholder="e.g. watercolor style, add sunset...", lines=2)
                strength_slider = gr.Slider(0.3, 0.9, value=0.6, step=0.1, label="Strength")
                modify_steps_slider = gr.Slider(10, 40, value=25, step=1, label="Steps")
                modify_guidance_slider = gr.Slider(5.0, 9.0, value=7.5, step=0.1, label="Guidance scale")
                modify_btn = gr.Button("Modify image", variant="secondary")

    PROMPT_PRESETS = {
        "arch": "Modern architecture, concrete and glass building, realistic, golden hour lighting, 8k, highly detailed, masterpiece",
        "interior": "Minimalist interior design, Scandinavian style, natural light, wooden floor, plants, cozy living room, 8k, ultra-detailed",
        "cyber": "Cyberpunk city, neon lights, rain-soaked streets, futuristic, cinematic lighting, blade runner style, 8k, highly detailed",
        "minimal": "Minimalist design, clean lines, monochrome, soft shadows, architectural photography, 8k, masterpiece",
        "anime": "Anime style, Makoto Shinkai, vibrant colors, detailed background, soft lighting, 8k, digital art",
        "photo": "Photorealistic, shallow depth of field, F1.8, natural lighting, professional photography, 8k, highly detailed",
        "watercolor": "Watercolor painting, soft edges, artistic, pastel colors, illustration style, hand-drawn feel, masterpiece",
        "fantasy": "Fantasy landscape, magical atmosphere, epic scale, dramatic lighting, ethereal, 8k, highly detailed, digital art",
    }
    btn_arch.click(fn=lambda: PROMPT_PRESETS["arch"], outputs=[prompt_txt])
    btn_interior.click(fn=lambda: PROMPT_PRESETS["interior"], outputs=[prompt_txt])
    btn_cyber.click(fn=lambda: PROMPT_PRESETS["cyber"], outputs=[prompt_txt])
    btn_minimal.click(fn=lambda: PROMPT_PRESETS["minimal"], outputs=[prompt_txt])
    btn_anime.click(fn=lambda: PROMPT_PRESETS["anime"], outputs=[prompt_txt])
    btn_photo.click(fn=lambda: PROMPT_PRESETS["photo"], outputs=[prompt_txt])
    btn_watercolor.click(fn=lambda: PROMPT_PRESETS["watercolor"], outputs=[prompt_txt])
    btn_fantasy.click(fn=lambda: PROMPT_PRESETS["fantasy"], outputs=[prompt_txt])

    run_btn.click(
        fn=process_sketch,
        inputs=[input_sketch, input_upload, prompt_txt, negative_prompt_txt, text_only_check, resolution_dropdown,
                num_steps_slider, guidance_scale_slider, controlnet_scale_slider, batch_size_input],
        outputs=[timer_lbl, output_gallery]
    )

    modify_btn.click(
        fn=process_modify,
        inputs=[modify_image_input, modify_prompt_txt, negative_prompt_txt, strength_slider, resolution_dropdown,
                modify_steps_slider, modify_guidance_slider],
        outputs=[timer_lbl, output_gallery]
    )

if __name__ == "__main__":
    demo.launch()
