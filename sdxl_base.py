"""
Chronora 時光 - 純 SDXL Base 1.0 版本
唔用 Lightning，畫質較好但速度較慢。步數 20-50、guidance_scale 約 7.5。
"""
import os
import warnings
# Suppress diffusers deprecation of upcast_vae (handled by library upgrade later)
warnings.filterwarnings("ignore", message=".*upcast_vae.*", category=FutureWarning)
import torch
import gradio as gr
from datetime import datetime
import time
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
try:
    import piexif
    from piexif import ExifIFD
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False
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
# 5. UI - Readable Light & Dark Theme
HEADING_AND_LABEL_CSS = """
/* Theme variables for accessibility (light by default) */
:root {
    --bg: #ffffff;
    --panel: #ffffff;
    --text: #0f172a;
    --muted: #374151;
    --accent: #3b82f6;
    --accent-strong: #2563eb;
    --input-bg: #f9fafb;
    --input-border: rgba(100, 150, 255, 0.28);
    --placeholder: #9ca3af;
    --card-shadow: 0 0 12px rgba(100, 150, 255, 0.08);
    --focus-glow: rgba(96,165,250,0.45);
}

/* Respect user's system preference for dark mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #071024;
        --panel: #0b1624;
        --text: #e6eef8;
        --muted: #cbd5e1;
        --accent: #60a5fa;
        --accent-strong: #1e90ff;
        --input-bg: #0b1220;
        --input-border: rgba(60, 90, 140, 0.36);
        --placeholder: #93a4b8;
        --card-shadow: 0 6px 20px rgba(2,6,23,0.6);
        --focus-glow: rgba(96,165,250,0.32);
    }
}

/* Manual dark override */
.gradio-container.dark, .gradio-container[data-theme="dark"] {
    --bg: #071024;
    --panel: #0b1624;
    --text: #e6eef8;
    --muted: #cbd5e1;
    --accent: #60a5fa;
    --accent-strong: #1e90ff;
    --input-bg: #0b1220;
    --input-border: rgba(60, 90, 140, 0.36);
    --placeholder: #93a4b8;
    --card-shadow: 0 6px 20px rgba(2,6,23,0.6);
    --focus-glow: rgba(96,165,250,0.32);
}

/* Apply base colors */
.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    min-height: 100vh !important;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
}

/* Panels */
.gradio-container .block, .gradio-container .form, .gradio-container .panel {
    background: var(--panel) !important;
    border-radius: 12px !important;
    border: 1px solid var(--input-border) !important;
    backdrop-filter: blur(6px) !important;
    box-shadow: var(--card-shadow) !important;
    padding: 12px !important;
}

/* Heading */
#app-heading, #app-heading p, #app-heading * {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    background: linear-gradient(135deg, rgba(219,234,254,0.65), rgba(224,242,254,0.65)) !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
    letter-spacing: 0.4px !important;
    display: block !important;
}

/* Labels */
.gradio-container label {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--accent-strong) !important;
    text-transform: capitalize !important;
}

/* Markdown text */
.gradio-container .markdown p {
    color: var(--muted) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

/* Inputs */
.gradio-container input, .gradio-container textarea, .gradio-container .gradio-input {
    background: var(--input-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-size: 1rem !important;
    transition: box-shadow 0.18s ease, border-color 0.18s ease !important;
}

.gradio-container input::placeholder, .gradio-container textarea::placeholder {
    color: var(--placeholder) !important;
}

.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container .gradio-input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 18px var(--focus-glow) !important;
    outline: none !important;
}

/* Dropdowns */
.gradio-container select, .gradio-container [data-testid="dropdown"], .gradio-container .gr-box {
    background: var(--input-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 1rem !important;
    transition: box-shadow 0.18s ease, border-color 0.18s ease !important;
}

.gradio-container select:hover, .gradio-container [data-testid="dropdown"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 12px var(--focus-glow) !important;
}

.gradio-container select:focus, .gradio-container [data-testid="dropdown"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px var(--focus-glow) !important;
    outline: none !important;
}

/* Dropdown menu & options (Gradio custom menus) */
.gradio-container .dropdown-menu {
    background: var(--panel) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15) !important;
}

.gradio-container [role="option"] {
    background: var(--panel) !important;
    color: var(--text) !important;
    padding: 10px 12px !important;
}
.gradio-container [role="option"]:hover, .gradio-container [role="option"][aria-selected="true"] {
    background: rgba(219,234,254,0.14) !important;
}

/* Buttons */
.gradio-container button[variant="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%) !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.12) !important;
}

.gradio-container button[variant="secondary"] {
    background: rgba(96,165,250,0.08) !important;
    color: var(--accent) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
}

/* Gallery */
#gallery, .gradio-container .gallery {
    border-radius: 8px !important;
    border: 1px solid var(--input-border) !important;
    background: var(--panel) !important;
}

/* Misc adjustments for readability */
.gradio-container .tab-nav button, .gradio-container label, .gradio-container .markdown h1, .gradio-container .markdown h2 {
    text-shadow: none !important;
}

.gradio-container *::-webkit-scrollbar { width: 8px !important; height: 8px !important; }
.gradio-container *::-webkit-scrollbar-track { background: transparent !important; }
.gradio-container *::-webkit-scrollbar-thumb { background: linear-gradient(180deg, rgba(100,150,255,0.3), rgba(96,165,250,0.5)) !important; border-radius: 4px !important; }
"""
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
    gr.Markdown("Chronora 時光 — SDXL Base 1.0", elem_id="app-heading")
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
            output_format_dropdown = gr.Dropdown(
                choices=OUTPUT_FORMAT_CHOICES,
                value="png",
                label="Output format"
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
                gr.Markdown("Steps 20–30 faster; 30–50 better quality. Output: `outputs_base/`")
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
                gr.Markdown("Upload image and enter a new prompt to vary it.")
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
                num_steps_slider, guidance_scale_slider, controlnet_scale_slider, batch_size_input, output_format_dropdown],
        outputs=[timer_lbl, output_gallery]
    )

    modify_btn.click(
        fn=process_modify,
        inputs=[modify_image_input, modify_prompt_txt, negative_prompt_txt, strength_slider, resolution_dropdown,
                modify_steps_slider, modify_guidance_slider, output_format_dropdown],
        outputs=[timer_lbl, output_gallery]
    )

if __name__ == "__main__":
    demo.launch()
