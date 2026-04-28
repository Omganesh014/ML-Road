# TruthBridge - Road Damage Detection System

Real-time road and bridge damage detection using deep learning. Detects and localizes surface damage (cracks, potholes) in infrastructure images.

## What It Does

**Input**: Image of road/bridge surface

**Output**:
- Bounding boxes around damage
- Damage classification (crack, pothole, no_damage)
- Confidence score
- Severity level (critical/moderate/minor)
- Risk score based on environmental factors

## Tech Stack

| Component | Technology |
|-----------|-------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| API Server | FastAPI |
| Database | SQLite |
| Image Hashing | imagehash |
| HTTP Client | httpx |

## Quick Start

```bash
# Clone repository
git clone https://github.com/Omganesh014/ML-Road.git
cd ML-Road

# Install dependencies
cd server && pip install -r requirements.txt && cd ..

# Initialize database
python server/init_db.py

# Run API server
cd server && uvicorn yolo_api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Upload image for damage detection |
| GET | `/health` | Check API and model status |
| GET | `/reports` | Get recent damage reports |
| GET | `/stats` | Get aggregate statistics |

### `/detect` Request

```
POST /detect
Content-Type: multipart/form-data

file: <image_file>
location: "bridge_12" (optional)
rainfall_mm: 45.2 (optional)
```

### `/detect` Response

```json
{
  "success": true,
  "duplicate": false,
  "image_hash": "abc123...",
  "detections": [
    {
      "damage_type": "pothole",
      "confidence": 85.2,
      "bbox": [120.5, 340.2, 450.1, 520.8],
      "coverage_percent": 45.3,
      "severity": "critical",
      "risk_score": 0.72
    }
  ],
  "model_info": {
    "model": "server/weights/best.pt",
    "device": "cuda",
    "num_detections": 1
  }
}
```

## System Architecture

```
Image Upload
    ↓
Compute imagehash.phash ← Duplicate Detection
    ↓
Load hash_db.json ← Check if already processed
    ↓
[Duplicate?] → YES → HTTP 409 Conflict
    ↓ NO
    ↓
YOLO Model (detect damage)
    ↓
Calculate Risk Score
    ↓
Store Report in SQLite
    ↓
[Severity = critical?] → YES → POST Webhook Alert
    ↓ NO
    ↓
Return Response
```

## Risk Score Formula

```
risk_score = (severity_weight × coverage_norm × location_risk) + (rainfall_mm × 0.01)
```

- **severity_weight**: critical=1.0, moderate=0.5, minor=0.2
- **coverage_norm**: bbox area / image area (0-1)
- **location_risk**: from database (default 0.5)
- **rainfall_mm**: environmental factor

## Database Schema

### damage_reports

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| image_hash | TEXT | Perceptual hash of uploaded image |
| damage_type | TEXT | crack/pothole/no_damage |
| confidence | REAL | Detection confidence % |
| bbox | TEXT | JSON [x1,y1,x2,y2] |
| coverage_percent | REAL | Area coverage % |
| severity | TEXT | critical/moderate/minor |
| location | TEXT | Location name |
| risk_score | REAL | Computed risk score |
| rainfall_mm | REAL | Rainfall at time of upload |
| created_at | TIMESTAMP | Upload timestamp |

### locations

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | Unique location identifier |
| base_risk | REAL | Base risk factor (0.0-1.0) |

## Project Structure

```
ML-Road/
├── data.yaml              # YOLO dataset config
├── yolo26n.pt           # Pretrained YOLOv8 nano
├── yolov8s.pt           # Pretrained YOLOv8 small
├── data/                # Raw datasets (not in git)
├── dataset_yolo/       # YOLO-formatted dataset (not in git)
├── models/
│   └── image_hashes.json  # Hash database
├── scripts/
│   ├── prepare_yolo_dataset.py  # Convert to YOLO format
│   ├── dedupe_dataset.py        # Build hash database
│   └── setup_ml.sh              # Setup script
└── server/
    ├── yolo_api.py     # FastAPI server
    ├── database.py     # SQLite operations
    ├── init_db.py      # Database initialization
    ├── config.yaml      # Configuration
    └── requirements.txt # Dependencies
```

## Configuration

Edit `server/config.yaml`:

```yaml
database:
  path: "damage_reports.db"

webhook:
  url: "http://localhost:5001/webhook"
  timeout: 10

hash_db:
  path: "../models/image_hashes.json"
  hamming_threshold: 10

severity_weights:
  critical: 1.0
  moderate: 0.5
  minor: 0.2

risk_factors:
  rainfall_multiplier: 0.01
  location_default: 0.5
```

## Webhook Integration

When critical damage is detected, a webhook is sent to the configured URL:

```json
{
  "alert": "critical_damage",
  "image_hash": "abc123...",
  "detections": [
    {
      "damage_type": "pothole",
      "confidence": 85.2,
      "coverage_percent": 45.3,
      "severity": "critical",
      "risk_score": 0.72
    }
  ],
  "location": "bridge_12",
  "rainfall_mm": 45.2,
  "max_risk_score": 0.72,
  "timestamp": "2026-04-29"
}
```

## Training Your Model

```bash
# Prepare dataset (if starting from raw images)
python scripts/prepare_yolo_dataset.py

# Build hash database
python scripts/dedupe_dataset.py

# Train YOLOv8
yolo detect train model=yolov8s.pt data=data.yaml epochs=50 imgsz=640

# Export trained model to server/weights/best.pt
```

## Model Capabilities

### What It DOES

- Detect damage in images
- Localize damage (bounding boxes)
- Classify damage type (crack, pothole, no_damage)
- Provide confidence scores
- Real-time inference (~3-10ms/image)
- Compute severity based on coverage
- Environmental risk scoring
- Duplicate image detection
- SQLite report storage
- Webhook alerts on critical damage

### What It Does NOT

- Detect AI-generated images
- Predict structural failure
- Detect all damage types (only cracks and potholes)

## Requirements

- Python 3.8+
- CUDA-capable GPU (for training)
- 8GB+ RAM
- ~5GB storage (for datasets and models)

## License

MIT
