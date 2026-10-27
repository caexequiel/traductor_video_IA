import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks
import torch

from app.state import app_state
from core.pipeline import DubbingPipeline

router = APIRouter(prefix="/api", tags=["process"])

class ProcessRequest(BaseModel):
    video_path: str
    subtitle_path: Optional[str] = None
    whisper_model: str = "medium"
    preserve_bg: bool = True
    source_language: Optional[str] = None
    dialect: str = "es-latam"

@router.get("/system/gpu")
def get_gpu_status():
    """Retorna información en tiempo real de la GPU y VRAM."""
    gpu_info = {
        "has_cuda": torch.cuda.is_available(),
        "device_name": "CPU",
        "vram_total_mb": 0,
        "vram_used_mb": 0,
        "vram_free_mb": 0,
        "temperature": 0
    }

    if torch.cuda.is_available():
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            if res.returncode == 0:
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 4:
                    gpu_info["vram_total_mb"] = int(parts[0])
                    gpu_info["vram_used_mb"] = int(parts[1])
                    gpu_info["vram_free_mb"] = int(parts[2])
                    gpu_info["temperature"] = int(parts[3])
        except Exception:
            # Fallback a métricas de torch
            tot = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            alloc = torch.cuda.memory_allocated(0) // (1024 * 1024)
            gpu_info["vram_total_mb"] = tot
            gpu_info["vram_used_mb"] = alloc
            gpu_info["vram_free_mb"] = tot - alloc

    return gpu_info

def run_pipeline_thread(req_dict: Dict[str, Any], job_id: str):
    pipeline = DubbingPipeline(work_dir="temp", output_dir="output")

    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def progress_callback(stage: str, percentage: float, message: str, extra: Dict[str, Any]):
        app_state.jobs[job_id]["stage"] = stage
        app_state.jobs[job_id]["progress"] = percentage
        app_state.jobs[job_id]["message"] = message
        if extra:
            app_state.jobs[job_id]["extra"].update(extra)

        # Broadcast via websocket
        update_msg = {
            "type": "job_update",
            "job_id": job_id,
            "stage": stage,
            "progress": percentage,
            "message": message,
            "extra": extra
        }
        loop.run_until_complete(app_state.broadcast(update_msg))

    try:
        app_state.jobs[job_id]["status"] = "running"
        result = pipeline.run(
            video_path=req_dict["video_path"],
            subtitle_path=req_dict.get("subtitle_path"),
            whisper_model=req_dict.get("whisper_model", "medium"),
            preserve_bg=req_dict.get("preserve_bg", True),
            source_language=req_dict.get("source_language"),
            dialect=req_dict.get("dialect", "es-latam"),
            progress_cb=progress_callback
        )
        app_state.jobs[job_id]["status"] = "completed"
        app_state.jobs[job_id]["result"] = result
        loop.run_until_complete(app_state.broadcast({
            "type": "job_completed",
            "job_id": job_id,
            "result": result
        }))
    except Exception as e:
        app_state.jobs[job_id]["status"] = "failed"
        app_state.jobs[job_id]["error"] = str(e)
        loop.run_until_complete(app_state.broadcast({
            "type": "job_failed",
            "job_id": job_id,
            "error": str(e)
        }))
    finally:
        loop.close()

@router.post("/process/start")
def start_process(req: ProcessRequest):
    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=400, detail="El archivo de video no existe.")

    import time
    job_id = f"job_{int(time.time())}"
    app_state.jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": "init",
        "progress": 0.0,
        "message": "En cola...",
        "extra": {},
        "req": req.dict()
    }

    t = threading.Thread(target=run_pipeline_thread, args=(req.dict(), job_id), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "started"}

@router.get("/process/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in app_state.jobs:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    return app_state.jobs[job_id]
