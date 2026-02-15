# Object Detection Service (CPU)

REST API service for object detection on images using FastAPI and YOLOv5.

## Features

- CPU-only inference
- Local model storage
- Model replacement via API
- Custom YOLOv5 model support
- Configurable inference size (320-1280)
- Adjustable bounding box thickness
- Batch image processing

## Quick Start

### 1. Prepare Model

Download YOLOv5 model and place it in `models/` folder:

```bash
mkdir -p models
# Download standard model
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O models/yolov5s.pt
```

Available models:
- `yolov5n.pt` - nano (fastest, least accurate)
- `yolov5s.pt` - small
- `yolov5m.pt` - medium
- `yolov5l.pt` - large
- `yolov5x.pt` - extra large (most accurate, slowest)

### 2. Run

**Local:**
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Docker:**
```bash
docker-compose up --build
```

## API Endpoints

### Detection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | JSON with detection results |
| `/detect/image` | POST | Image with bounding boxes |
| `/detect/batch` | POST | Batch processing |
| `/classes` | GET | Model class list |
| `/sizes` | GET | Available inference sizes |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models` | GET | List all models |
| `/models/current` | GET | Active model info |
| `/models/upload` | POST | Upload new model |
| `/models/activate/{name}` | POST | Activate model |
| `/models/{name}` | DELETE | Delete model |

## Parameters

### Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_threshold` | float | 0.25 | Confidence threshold (0-1) |
| `imgsz` | string | "640" | Inference image size |
| `line_thickness` | int | 2 | Bounding box thickness (1-10) |

### Available Image Sizes

| Size | Description |
|------|-------------|
| 320 | Fast, low quality |
| 416 | Fast |
| 512 | Medium |
| 640 | Standard (default) |
| 768 | High quality |
| 1024 | Very high quality |
| 1280 | Maximum quality, slow |

## Usage Examples

### Upload New Model

```bash
# Upload and activate immediately
curl -X POST "http://localhost:8000/models/upload?activate=true" \
  -F "file=@yolov5m.pt"

# Upload only (without activation)
curl -X POST "http://localhost:8000/models/upload?activate=false" \
  -F "file=@yolov5l.pt"
```

### Switch Between Models

```bash
# List models
curl http://localhost:8000/models

# Activate another model
curl -X POST "http://localhost:8000/models/activate/yolov5m.pt"

# Check current model
curl http://localhost:8000/models/current
```

### Object Detection

```bash
# Get JSON result
curl -X POST "http://localhost:8000/detect?imgsz=640&confidence_threshold=0.5" \
  -F "file=@image.jpg"

# Get image with bounding boxes
curl -X POST "http://localhost:8000/detect/image?imgsz=768&line_thickness=3" \
  -F "file=@image.jpg" \
  --output result.jpg
```

### Python Client

```python
import requests

API = "http://localhost:8000"

# Upload model
with open("my_custom_model.pt", "rb") as f:
    resp = requests.post(
        f"{API}/models/upload",
        files={"file": f},
        params={"activate": "true"}
    )
    print(resp.json())

# Detection
with open("image.jpg", "rb") as f:
    resp = requests.post(
        f"{API}/detect",
        files={"file": f},
        params={
            "confidence_threshold": 0.5,
            "imgsz": "640"
        }
    )
    print(resp.json())

# Get image with boxes
with open("image.jpg", "rb") as f:
    resp = requests.post(
        f"{API}/detect/image",
        files={"file": f},
        params={
            "confidence_threshold": 0.5,
            "imgsz": "1024",
            "line_thickness": 3
        }
    )
    with open("result.jpg", "wb") as out:
        out.write(resp.content)
```

## Project Structure

```
object_detection_service/
├── main.py              # FastAPI application
├── requirements.txt     # Dependencies
├── Dockerfile
├── docker-compose.yml
├── client.py            # Example client
├── models/              # Models directory
│   ├── yolov5s.pt       # YOLOv5 model
│   └── active_model.pt  # Active model copy
├── README.md            # English documentation
└── README_UA.md         # Ukrainian documentation
```

## Custom Models

Service supports custom YOLOv5 models trained on your data.

**Model requirements:**
- PyTorch format (.pt)
- YOLOv5 architecture
- Compatible with `torch.hub.load("ultralytics/yolov5", "custom", ...)`

**Upload custom model:**
```bash
curl -X POST "http://localhost:8000/models/upload" \
  -F "file=@my_custom_model.pt" \
  -F "activate=true"
```

## Performance (CPU)

| Model | Size | Inference Time* |
|-------|------|-----------------|
| yolov5n | 4 MB | ~50ms |
| yolov5s | 14 MB | ~100ms |
| yolov5m | 42 MB | ~200ms |
| yolov5l | 92 MB | ~350ms |
| yolov5x | 174 MB | ~500ms |

*Intel Core i7, 640x640 image

## API Documentation

After starting the server, interactive documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT
