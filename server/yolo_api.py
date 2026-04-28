"""
TruthBridge YOLO Damage Detection API
FastAPI server for YOLOv8 damage detection with duplicate detection,
risk scoring, and webhook alerts.
Run with: uvicorn server.yolo_api:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import json
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

import imagehash
import yaml
import httpx
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

app = FastAPI(title="TruthBridge Damage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "weights" / "best.pt"
model = None
config = None
db = None

CLASS_LABELS = {
    0: "crack",
    1: "pothole",
    2: "no_damage"
}


def load_config():
    global config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def compute_image_hash(img_bytes: bytes) -> Optional[str]:
    try:
        img = Image.open(io.BytesIO(img_bytes))
        h = imagehash.phash(img, hash_size=16)
        return str(h)
    except Exception:
        return None


def check_duplicate(img_hash: str, hash_db: dict, threshold: int) -> bool:
    if not hash_db or "hashes" not in hash_db:
        return False

    for entry in hash_db["hashes"]:
        stored_hash = entry["hash"]
        dist = imagehash.hex_to_hash(img_hash) - imagehash.hex_to_hash(stored_hash)
        if dist <= threshold:
            return True
    return False


def compute_risk_score(
    severity: str,
    coverage_percent: float,
    location: Optional[str],
    rainfall_mm: float,
    location_risk: float
) -> float:
    severity_weights = config["severity_weights"]
    rainfall_multiplier = config["risk_factors"]["rainfall_multiplier"]

    severity_weight = severity_weights.get(severity, 0.2)
    coverage_norm = coverage_percent / 100.0
    location_risk_val = location_risk if location_risk is not None else config["risk_factors"]["location_default"]
    rainfall_factor = rainfall_mm * rainfall_multiplier

    risk = (severity_weight * coverage_norm * location_risk_val) + rainfall_factor
    return min(max(risk, 0.0), 1.0)


async def send_webhook(payload: dict):
    webhook_config = config["webhook"]
    url = webhook_config["url"]
    timeout = webhook_config["timeout"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=timeout
            )
            if response.status_code >= 200 and response.status_code < 300:
                print(f"Webhook sent successfully to {url}")
                return True
            else:
                print(f"Webhook failed with status {response.status_code}")
                return False
    except Exception as e:
        print(f"Webhook error: {e}")
        return False


class DetectionResult(BaseModel):
    damage_type: str
    confidence: float
    bbox: list
    coverage_percent: float
    severity: str
    risk_score: Optional[float] = None


class YOLOResponse(BaseModel):
    success: bool
    duplicate: bool = False
    detections: list[DetectionResult]
    image_hash: Optional[str] = None
    model_info: dict


def load_model():
    global model
    if not ULTRALYTICS_AVAILABLE:
        print("⚠️ ultralytics not installed. Run: pip install ultralytics")
        return None

    if not MODEL_PATH.exists():
        print(f"⚠️ Model not found at {MODEL_PATH}")
        print("   Run training first, then export model to server/weights/best.pt")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO model on {device}...")
    model = YOLO(str(MODEL_PATH))
    model.to(device)
    print("✅ YOLO model loaded")
    return model


def compute_severity(coverage_percent: float) -> str:
    thresholds = {
        "critical": 0.40,
        "moderate": 0.15,
        "minor": 0.0
    }
    if coverage_percent >= thresholds["critical"]:
        return "critical"
    elif coverage_percent >= thresholds["moderate"]:
        return "moderate"
    return "minor"


@app.on_event("startup")
async def startup():
    global db
    load_config()

    from database import Database
    db = Database(config["database"]["path"])
    await db.init_db()

    load_model()


@app.post("/detect", response_model=YOLOResponse)
async def detect_damage(
    file: UploadFile = File(...),
    location: Optional[str] = Form(None),
    rainfall_mm: Optional[float] = Form(0.0),
):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and place model at server/weights/best.pt")

    contents = await file.read()

    img_hash = compute_image_hash(contents)
    if img_hash and config.get("hash_db"):
        hash_db_path = Path(__file__).parent / config["hash_db"]["path"]
        if hash_db_path.exists():
            with open(hash_db_path) as f:
                hash_db = json.load(f)
            threshold = config["hash_db"].get("hamming_threshold", 10)
            if check_duplicate(img_hash, hash_db, threshold):
                return YOLOResponse(
                    success=False,
                    duplicate=True,
                    detections=[],
                    image_hash=img_hash,
                    model_info={"reason": "Duplicate image detected"}
                )

    results = model(contents, imgsz=640, conf=0.25, verbose=False)

    detections = []
    max_severity = "minor"
    max_risk = 0.0

    location_risk = None
    if location:
        location_risk = await db.get_location_risk(location)

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            detections.append(DetectionResult(
                damage_type="no_damage",
                confidence=0.0,
                bbox=[],
                coverage_percent=0.0,
                severity="minor",
                risk_score=0.0
            ))
            continue

        img_area = result.orig_shape[0] * result.orig_shape[1]

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            x1, y1, x2, y2 = xyxy
            bbox_area = (x2 - x1) * (y2 - y1)
            coverage = (bbox_area / img_area) * 100

            severity = compute_severity(coverage)
            risk_score = compute_risk_score(severity, coverage, location, rainfall_mm, location_risk)

            if severity == "critical" or risk_score > max_risk:
                max_severity = severity if severity == "critical" else max_severity
                max_risk = risk_score

            detections.append(DetectionResult(
                damage_type=CLASS_LABELS.get(cls_id, "unknown"),
                confidence=round(conf * 100, 1),
                bbox=[round(x, 2) for x in xyxy],
                coverage_percent=round(coverage, 2),
                severity=severity,
                risk_score=round(risk_score, 3)
            ))

    if detections and not detections[0].damage_type == "no_damage":
        for det in detections:
            await db.insert_report(
                image_hash=img_hash or "unknown",
                damage_type=det.damage_type,
                confidence=det.confidence,
                bbox=det.bbox,
                coverage_percent=det.coverage_percent,
                severity=det.severity,
                location=location,
                risk_score=det.risk_score,
                rainfall_mm=rainfall_mm,
            )

        if max_severity == "critical":
            webhook_payload = {
                "alert": "critical_damage",
                "image_hash": img_hash,
                "detections": [
                    {
                        "damage_type": d.damage_type,
                        "confidence": d.confidence,
                        "coverage_percent": d.coverage_percent,
                        "severity": d.severity,
                        "risk_score": d.risk_score,
                    }
                    for d in detections
                ],
                "location": location,
                "rainfall_mm": rainfall_mm,
                "max_risk_score": round(max_risk, 3),
                "timestamp": result.orig_shape if hasattr(result, 'orig_shape') else None,
            }
            await send_webhook(webhook_payload)

    return YOLOResponse(
        success=True,
        duplicate=False,
        detections=detections,
        image_hash=img_hash,
        model_info={
            "model": str(MODEL_PATH),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_detections": len(detections),
            "max_severity": max_severity,
        }
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.get("/reports")
async def get_reports(location: Optional[str] = None, limit: int = 10):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    reports = await db.get_recent_reports(location=location, limit=limit)
    return {"reports": reports}


@app.get("/stats")
async def get_stats():
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    stats = await db.get_stats()
    return stats


@app.get("/")
async def root():
    return {
        "message": "TruthBridge Damage Detection API",
        "endpoints": {
            "POST /detect": "Upload image for damage detection",
            "GET /health": "Check API health",
            "GET /reports": "Get recent damage reports",
            "GET /stats": "Get aggregate statistics",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)