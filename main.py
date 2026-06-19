from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(title="Multimodal Fusion API", description="Centralized backend for motion and audio integration.")

# --- 1. THE API CONTRACTS (The PM Rules) ---
class VisionInput(BaseModel):
    timestamp: str
    motion_detected: bool
    confidence_score: float
    bounding_box: List[int] # Expects [x, y, width, height]

class AudioInput(BaseModel):
    timestamp: str
    noise_level_db: float
    is_human_voice: bool

class FusedOutput(BaseModel):
    status: str
    threat_level: str
    latency_ms: float

# --- 2. THE FUSION ENDPOINT ---
@app.post("/api/v1/fuse", response_model=FusedOutput)
async def fuse_multimodal_data(vision: VisionInput, audio: Optional[AudioInput] = None):
    start_time = time.time()
    
    # Simulating a rejection if the Vision model sends bad data
    if vision.confidence_score < 0.0 or vision.confidence_score > 1.0:
        raise HTTPException(status_code=400, detail="Invalid confidence score. Must be between 0 and 1.")

    # Feature Fusion Logic
    threat = "Low"
    if vision.motion_detected and vision.confidence_score > 0.8:
        threat = "Medium"
        # If audio is present and confirms a human voice, escalate threat
        if audio and audio.is_human_voice and audio.noise_level_db > 60:
            threat = "High"

    end_time = time.time()
    latency = round((end_time - start_time) * 1000, 2)

    return FusedOutput(
        status="Success",
        threat_level=threat,
        latency_ms=latency
    )