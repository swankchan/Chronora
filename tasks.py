import os
import time
from datetime import datetime
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
try:
    import piexif
    from piexif import ExifIFD
    HAS_PIEXIF = True
except Exception:
    HAS_PIEXIF = False

from diffusers import StableDiffusionXLPipeline
from rq import get_current_job

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_base")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pipeline singleton
_pipe = None
_device = "cpu"
_dtype = torch.float32

def init_pipeline(device=None):
    global _pipe, _device, _dtype
    if _pipe is not None:
        return _pipe
    has_acc = torch.cuda.is_available() if device is None else (device == "cuda")
    _device = "cuda" if has_acc else "cpu"
    _dtype = torch.float16 if has_acc else torch.float32
    # Load base pipeline
    _pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=_dtype,
        variant="fp16" if has_acc else None,
    )
    try:
        if has_acc:
            _pipe.enable_model_cpu_offload()
        else:
            _pipe = _pipe.to(_device)
    except Exception:
        _pipe = _pipe.to(_device)
    _pipe.enable_attention_slicing()
    return _pipe


def _metadata_comment(prompt, negative_prompt, model_id="stabilityai/stable-diffusion-xl-base-1.0", extra=None):
    parts = [
        f"prompt: {prompt or ''}",
        f"negative_prompt: {negative_prompt or ''}",
        f"model: {model_id}",
    ]
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def _jpeg_exif_bytes_with_comment(comment):
    if not HAS_PIEXIF:
        return None
    try:
        if hasattr(piexif, "helper") and hasattr(piexif.helper, "UserComment"):
            user_comment = piexif.helper.UserComment.dump(comment, encoding="unicode")
        else:
            user_comment = b"ASCII\x00\x00\x00" + comment.encode("ascii", errors="replace")
        exif_dict = {
            "0th": {},
            "Exif": {ExifIFD.UserComment: user_comment},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        return piexif.dump(exif_dict)
    except Exception:
        return None


def save_image_with_metadata(pil_image: Image.Image, save_path: str, prompt: str, negative_prompt: str, extra: str = None, output_ext: str = "png"):
    comment = _metadata_comment(prompt, negative_prompt, extra=extra)
    img = pil_image.convert("RGB")
    ext = output_ext.lower()
    if ext == "png":
        pnginfo = PngInfo()
        pnginfo.add_text("Comment", comment)
        pnginfo.add_text("parameters", comment)
        img.save(save_path, pnginfo=pnginfo)
    elif ext in ("jpg", "jpeg"):
        exif_bytes = _jpeg_exif_bytes_with_comment(comment)
        save_kw = {"quality": 100, "comment": comment}
        if exif_bytes:
            save_kw["exif"] = exif_bytes
        img.save(save_path, "JPEG", **save_kw)
    elif ext == "webp":
        img.save(save_path, "WEBP", quality=90)
    else:
        img.save(save_path)


def generate_text2img(prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, steps: int = 30, guidance_scale: float = 7.5, batch_size: int = 1, output_format: str = "png"):
    """Synchronous generation function intended to be run by background worker.
    Returns list of saved absolute file paths.
    """
    pipe = init_pipeline()
    outputs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    gen_kwargs = {
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "height": int(height),
        "width": int(width),
    }
    if negative_prompt and negative_prompt.strip():
        gen_kwargs["negative_prompt"] = negative_prompt.strip()

    for i in range(int(batch_size)):
        result = pipe(prompt, **gen_kwargs)
        img = result.images[0]
        ext = (output_format or "png").strip().lower()
        if ext not in ("png", "jpg", "jpeg", "webp"):
            ext = "png"
        if batch_size > 1:
            filename = f"chronora_base_{timestamp}_{i+1:03d}.{ext}"
        else:
            filename = f"chronora_base_{timestamp}.{ext}"
        save_path = os.path.join(OUTPUT_DIR, filename)
        extra = f"steps: {int(steps)} | guidance_scale: {float(guidance_scale)}"
        save_image_with_metadata(img, save_path, prompt, negative_prompt or "", extra=extra, output_ext=ext)
        outputs.append(save_path)

        # Update RQ job meta so API/frontend can observe partial results progressively
        try:
            job = get_current_job()
            if job is not None:
                files_meta = job.meta.get("files", [])
                files_meta.append(save_path)
                job.meta["files"] = files_meta
                # keep batch size and completed count in meta
                job.meta.setdefault("batch_size", int(batch_size))
                job.meta["completed"] = len(files_meta)
                job.save_meta()
        except Exception:
            # non-fatal: continue generating
            pass

    return outputs
