from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(
    title="Multimodal Fusion API", 
    description="Centralized backend for motion and audio integration.",
    openapi_url="/openapi.json",
    docs_url="/docs"
)

# PM Fix: Allow Render to bypass CORS blocks for the API Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "System Online", "message": "Navigate to /docs to view the interactive dashboard."}

class VisionInput(BaseModel):
    timestamp: str
    motion_detected: bool
    confidence_score: float
    bounding_box: List[int]

class AudioInput(BaseModel):
    timestamp: str
    noise_level_db: float
    is_human_voice: bool

class FusedOutput(BaseModel):
    status: str
    threat_level: str
    latency_ms: float

@app.post("/api/v1/fuse", response_model=FusedOutput)
async def fuse_multimodal_data(vision: VisionInput, audio: Optional[AudioInput] = None):
    start_time = time.time()
    
    if vision.confidence_score < 0.0 or vision.confidence_score > 1.0:
        raise HTTPException(status_code=400, detail="Invalid confidence score. Must be between 0 and 1.")

    threat = "Low"
    if vision.motion_detected and vision.confidence_score > 0.8:
        threat = "Medium"
        if audio and audio.is_human_voice and audio.noise_level_db > 60:
            threat = "High"

    end_time = time.time()
    latency = round((end_time - start_time) * 1000, 2)

    return FusedOutput(status="Success", threat_level=threat, latency_ms=latency)
