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
except (ImportError, AttributeError) as e:
    print(f"⚠️ HEDdetector 載入失敗: {e}")
    try:
        from controlnet_aux.hed import HEDdetector
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        print("✅ HEDdetector (替代路徑) 載入成功")
    except Exception as e2:
        print(f"❌ HEDdetector 載入完全失敗: {e2}")
        hed = None

# 3. 純 SDXL Base 1.0（唔載入 Lightning）
dtype = torch.float16 if has_accelerator else torch.float32

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=dtype
)

# 只載入一次：主 pipeline 用純 Base 1.0
print("📦 載入 SDXL Base 1.0（純 Base，唔用 Lightning）...")
pipe_t2i = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
    variant="fp16" if has_accelerator else None
)
print("✅ SDXL Base 1.0 載入成功！")

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
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_base")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

OUTPUT_FORMAT_CHOICES = [
    ("PNG (lossless, stores prompt)", "png"),
    ("JPEG (quality 100, EXIF Comment)", "jpeg"),
    ("JPEG 2000 (lossless .jp2, comment in .jp2.txt)", "jp2"),
    ("WebP (quality 90)", "webp"),
]

def _metadata_comment(prompt, negative_prompt, model_id, extra):
    """統一代碼：生成 prompt/negative/model 嘅 comment 字串。"""
    parts = [
        f"prompt: {prompt or ''}",
        f"negative_prompt: {negative_prompt or ''}",
        f"model: {model_id}",
    ]
    if extra:
        parts.append(extra)
    return "\n".join(parts)

def _jpeg_exif_bytes_with_comment(comment):
    """Build EXIF UserComment bytes for JPEG. Uses piexif.helper if available, else manual ASCII encoding."""
    if not HAS_PIEXIF:
        return None
    try:
        if hasattr(piexif, "helper") and hasattr(piexif.helper, "UserComment"):
            user_comment = piexif.helper.UserComment.dump(comment, encoding="unicode")
        else:
            # Older piexif without helper: EXIF UserComment = 8-byte charset + data (ASCII)
            user_comment = b"ASCII\x00\x00\x00" + comment.encode("ascii", errors="replace")
        exif_dict = {
            "0th": {},
            "Exif": {ExifIFD.UserComment: user_comment},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        return piexif.dump(exif_dict)
    except Exception as e:
        print(f"JPEG EXIF failed (using COM comment only): {e}")
        return None

def save_image_with_metadata(pil_image, save_path, prompt, negative_prompt, model_id=MODEL_ID, extra=None, output_ext="png"):
    """按 output_ext 儲存為 PNG / JPEG / WebP；PNG 寫 Comment，JPEG 寫 EXIF UserComment。"""
    comment = _metadata_comment(prompt, negative_prompt, model_id, extra)
    img = pil_image.convert("RGB")
    ext = output_ext.lower()
    if ext == "png":
        pnginfo = PngInfo()
        pnginfo.add_text("Comment", comment)
        pnginfo.add_text("parameters", comment)
        img.save(save_path, pnginfo=pnginfo)
    elif ext in ("jpg", "jpeg"):
        exif_bytes = _jpeg_exif_bytes_with_comment(comment)
        # 一定用 comment= 寫入 JPEG COM segment，exiftool 會顯示為 Comment
        save_kw = {"quality": 100, "comment": comment}
        if exif_bytes:
            save_kw["exif"] = exif_bytes
        img.save(save_path, "JPEG", **save_kw)
    elif ext == "jp2":
        # JPEG 2000 無損；Pillow 的 JP2 不支援 exif=，改寫 sidecar .txt 存 comment
        try:
            img.save(save_path, "JPEG2000", lossless=True)
            sidecar_path = save_path + ".txt"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.write(comment)
        except Exception:
            img.save(save_path, "PNG")  # fallback
    elif ext == "webp":
        img.save(save_path, "WEBP", quality=90)
    else:
        img.save(save_path)

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
                   num_steps, guidance_scale, controlnet_scale, batch_size, output_format):
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

    # 純 Base 1.0 參數：用 guidance_scale，步數 20-50
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

    print(f"🔄 開始批量生成 {batch_size} 張圖片（Base 1.0）...")
    for i in range(batch_size):
        print(f"   生成第 {i+1}/{batch_size} 張...")

        if composite_img is None:
            result = pipe_t2i(prompt, **gen_kwargs)
        else:
            result = pipe(prompt, **gen_kwargs)

        output = result.images[0]
        outputs.append(output)

        ext = (output_format or "png").strip().lower()
        if ext not in ("png", "jpg", "jpeg", "jp2", "webp"):
            ext = "png"
        if batch_size > 1:
            filename = f"chronora_base_{timestamp}_{i+1:03d}.{ext}"
        else:
            filename = f"chronora_base_{timestamp}.{ext}"
        save_path = os.path.join(OUTPUT_DIR, filename)
        extra = f"steps: {int(num_steps)} | guidance_scale: {float(guidance_scale)}"
        save_image_with_metadata(output, save_path, prompt, negative_prompt or "", model_id=MODEL_ID, extra=extra, output_ext=ext)
        saved_files.append(filename)

        if has_accelerator and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    end_time = time.time()
    total_ms = (end_time - start_time) * 1000
    total_sec = total_ms / 1000
    avg_ms = total_ms / batch_size
    mode = "Text only" if composite_img is None else "Sketch+text"
    quality_info = f"steps:{int(num_steps)} | guidance:{float(guidance_scale):.1f} | Base 1.0"

    if batch_size > 1:
        status_msg = f"Done: {mode} | {batch_size} images | {quality_info} | {total_sec:.1f}s total | {avg_ms:.0f} ms/img | Saved: {saved_files[0]} (+{batch_size} files)"
    else:
        status_msg = f"Done: {mode} | {quality_info} | {total_ms:.2f} ms | Saved: {saved_files[0]}"
    return status_msg, outputs

def process_modify(init_image, modify_prompt, negative_prompt, strength, resolution_choice, num_steps, guidance_scale, output_format):
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
        "guidance_scale": float(guidance_scale)  # Base 1.0 用返 guidance
    }
    if negative_prompt and negative_prompt.strip():
        kwargs["negative_prompt"] = negative_prompt.strip()
    output = p(**kwargs).images[0]

    ext = (output_format or "png").strip().lower()
    if ext not in ("png", "jpg", "jpeg", "jp2", "webp"):
        ext = "png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chronora_base_mod_{timestamp}.{ext}"
    save_path = os.path.join(OUTPUT_DIR, filename)
    extra = f"steps: {int(num_steps)} | guidance_scale: {float(guidance_scale)} | strength: {float(strength)}"
    save_image_with_metadata(output, save_path, modify_prompt.strip(), negative_prompt or "", model_id=MODEL_ID, extra=extra, output_ext=ext)

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
