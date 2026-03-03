import os
import math
import time
from typing import Optional
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import redis
from rq import Queue
from dotenv import load_dotenv

# Load .env file (if present) for local development
load_dotenv()

# Use default outputs directory from existing code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_base")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Redis + RQ queue (configurable timeouts)
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# Socket timeout for redis client (seconds, float)
REDIS_SOCKET_TIMEOUT = float(os.environ.get("REDIS_SOCKET_TIMEOUT", "5"))
# RQ timeouts (seconds)
RQ_DEFAULT_TIMEOUT = int(os.environ.get("RQ_DEFAULT_TIMEOUT", "21600"))
# Per-job timeout used when enqueueing (falls back to RQ_DEFAULT_TIMEOUT)
RQ_JOB_TIMEOUT = int(os.environ.get("RQ_JOB_TIMEOUT", str(RQ_DEFAULT_TIMEOUT)))

redis_conn = redis.from_url(redis_url, socket_timeout=REDIS_SOCKET_TIMEOUT)
queue = Queue("default", connection=redis_conn, default_timeout=RQ_DEFAULT_TIMEOUT)

print(f"[sdxl_direct] Redis: {redis_url} socket_timeout={REDIS_SOCKET_TIMEOUT}; RQ default_timeout={RQ_DEFAULT_TIMEOUT}")

app = FastAPI(title="Chronora SDXL Direct API")

# Allow frontend dev server (Vite) to reach this API during development
FRONTEND_ORIGINS = os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved outputs
app.mount("/outputs_base", StaticFiles(directory=OUTPUT_DIR), name="outputs_base")

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/generate")
async def api_generate(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    resolution: str = Form("1024x1024"),
    steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    batch_size: int = Form(1),
    output_format: str = Form("png"),
):
    """Enqueue a generation job. Returns job_id for polling."""
    if not prompt or prompt.strip() == "":
        raise HTTPException(status_code=400, detail="prompt is required")

    # validate basic limits
    if batch_size < 1 or batch_size > 10:
        raise HTTPException(status_code=400, detail="batch_size must be 1..10")

    try:
        w, h = resolution.lower().split("x")
        width, height = int(w), int(h)
    except Exception:
        raise HTTPException(status_code=400, detail="resolution must be like 1024x1024")

    if width * height > 4096 * 4096:
        raise HTTPException(status_code=400, detail="requested resolution too large")

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "width": width,
        "height": height,
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "batch_size": int(batch_size),
        "output_format": output_format.lower(),
    }

    # enqueue task (tasks.generate_text2img will be executed by worker)
    try:
        job = queue.enqueue("tasks.generate_text2img", kwargs=payload, timeout=RQ_JOB_TIMEOUT)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

    return JSONResponse(status_code=202, content={
        "status": "accepted",
        "job_id": job.id,
        "poll_url": f"/api/v1/job/{job.id}"
    })

@app.get("/api/v1/job/{job_id}")
def job_status(job_id: str):
    from rq.job import Job
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="job not found")

    resp = {
        "id": job.id,
        "status": job.get_status(),
        "result": None,
        "meta": job.meta,
    }

    # Provide partial results from job.meta if present (files produced so far)
    urls = []
    # prefer final result list when finished
    if job.is_finished:
        files = job.result or []
    else:
        files = job.meta.get("files", []) or []

    for p in files:
        fname = os.path.basename(p)
        urls.append({
            "filename": fname,
            "url": f"/outputs_base/{fname}"
        })

    if urls:
        resp["result"] = {"files": urls}
    else:
        resp["result"] = None

    if job.is_failed:
        resp["error"] = str(job.exc_info)

    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sdxl_direct:app", host="0.0.0.0", port=8000, reload=False)
