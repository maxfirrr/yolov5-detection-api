"""
Object Detection Service using FastAPI and YOLOv5 (Local Model, CPU Only)
"""

import io
import os
import shutil
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.openapi.docs import get_redoc_html
from PIL import Image
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel
from enum import Enum
import logging
import cv2
import numpy as np

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageSize(str, Enum):
    """Available image sizes for inference"""
    SIZE_320 = "320"    # Fast, low quality
    SIZE_416 = "416"    # Fast
    SIZE_512 = "512"    # Medium
    SIZE_640 = "640"    # Standard (default)
    SIZE_768 = "768"    # High quality
    SIZE_1024 = "1024"  # Very high quality
    SIZE_1280 = "1280"  # Maximum quality, slow


# Size mapping
IMAGE_SIZES = {
    "320": (320, 320),
    "416": (416, 416),
    "512": (512, 512),
    "640": (640, 640),
    "768": (768, 768),
    "1024": (1024, 1024),
    "1280": (1280, 1280),
}


def draw_boxes(image: np.ndarray, results, line_thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image with specified line thickness.
    
    Args:
        image: Image as numpy array (RGB)
        results: YOLOv5 detection results
        line_thickness: Line thickness in pixels
    
    Returns:
        Image with drawn bounding boxes
    """
    img = image.copy()
    predictions = results.pandas().xyxy[0]
    names = results.names
    
    for _, row in predictions.iterrows():
        x1, y1 = int(row['xmin']), int(row['ymin'])
        x2, y2 = int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = int(row['class'])
        label = f"{names[cls]} {conf:.2f}"
        
        # Color for class (simple hash function for different colors)
        color = (
            (cls * 50) % 255,
            (cls * 80 + 100) % 255,
            (cls * 120 + 50) % 255
        )
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=line_thickness)
        
        # Font scale proportional to line thickness
        font_scale = line_thickness / 3
        font_thickness = max(1, line_thickness - 1)
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Draw text background
        cv2.rectangle(
            img, 
            (x1, y1 - text_height - baseline - 5), 
            (x1 + text_width, y1), 
            color, 
            -1  # Fill
        )
        
        # Draw text
        cv2.putText(
            img, 
            label, 
            (x1, y1 - baseline - 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 255),  # White text
            font_thickness,
            cv2.LINE_AA
        )
    
    return img


# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = "cpu"

# Fix for Windows: PosixPath -> WindowsPath
# Required for loading models saved on Linux
import pathlib
import platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Model directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Default active model name
DEFAULT_MODEL_NAME = "yolov5s.pt"
ACTIVE_MODEL_PATH = MODELS_DIR / "active_model.pt"

app = FastAPI(
    title="Object Detection API",
    description="API for object detection on images using YOLOv5 (CPU)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,  # Disable default, use custom
    openapi_url="/openapi.json"
)


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc with alternative CDN"""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://unpkg.com/redoc@latest/bundles/redoc.standalone.js"
    )

# Global variable for model
model = None
current_model_name = None


class DetectionResult(BaseModel):
    """Detection result model"""
    class_name: str
    confidence: float
    bbox: list[float]


class DetectionResponse(BaseModel):
    """API response model"""
    success: bool
    detections: list[DetectionResult]
    total_objects: int
    model_name: str


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    size_mb: float
    is_active: bool


def load_model(model_path: Path) -> bool:
    """Load local YOLOv5 model"""
    global model, current_model_name
    
    try:
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        logger.info(f"Loading model: {model_path}")
        
        # Save original torch.load function
        original_load = torch.load
        
        # Override torch.load to always use weights_only=False
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        # Apply patch
        torch.load = patched_load
        
        try:
            # Load custom model using YOLOv5
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(model_path),
                device=DEVICE,
                force_reload=False,
                trust_repo=True
            )
        finally:
            # Restore original function
            torch.load = original_load
        
        # Force CPU mode
        model.to(DEVICE)
        model.eval()
        
        current_model_name = model_path.name
        logger.info(f"Model loaded successfully: {current_model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return False


def validate_model(model_path: Path) -> bool:
    """Validate YOLOv5 model"""
    try:
        import pathlib
        import platform
        
        # Fix for Windows: PosixPath -> WindowsPath
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        
        # Try to load model directly via torch
        # with weights_only=False for YOLOv5 compatibility
        checkpoint = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
        
        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            # Standard YOLOv5 format
            if 'model' in checkpoint or 'ema' in checkpoint:
                return True
        
        # Could be just state_dict or entire model
        return True
        
    except Exception as e:
        logger.error(f"Model validation error: {e}")
        return False


def download_default_model():
    """Download default YOLOv5s model if no active model exists"""
    default_path = MODELS_DIR / DEFAULT_MODEL_NAME
    
    if not default_path.exists() and not ACTIVE_MODEL_PATH.exists():
        logger.info("Downloading default YOLOv5s model...")
        try:
            # Download .pt file directly
            import urllib.request
            url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
            urllib.request.urlretrieve(url, default_path)
            
            logger.info(f"Model saved: {default_path}")
            
            # Copy as active model
            shutil.copy(default_path, ACTIVE_MODEL_PATH)
            
        except Exception as e:
            logger.error(f"Model download error: {e}")
            raise


@app.on_event("startup")
async def startup_event():
    """Load model on server startup"""
    logger.info("Starting Object Detection Service (CPU mode)...")
    logger.info(f"Models directory: {MODELS_DIR}")
    
    # Check for active model
    if ACTIVE_MODEL_PATH.exists():
        success = load_model(ACTIVE_MODEL_PATH)
        if success:
            return
    
    # Check for models in directory
    model_files = list(MODELS_DIR.glob("*.pt"))
    if model_files:
        # Use first found model
        success = load_model(model_files[0])
        if success:
            shutil.copy(model_files[0], ACTIVE_MODEL_PATH)
            return
    
    # Download default model
    try:
        download_default_model()
        load_model(ACTIVE_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")
        logger.warning("Upload model via POST /models/upload")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Object Detection API (CPU)",
        "docs": "/docs",
        "current_model": current_model_name,
        "device": DEVICE,
        "endpoints": {
            "/detect": "POST - Object detection",
            "/detect/image": "POST - Image with bounding boxes",
            "/models": "GET - List models",
            "/models/upload": "POST - Upload new model",
            "/models/activate/{name}": "POST - Activate model",
            "/models/{name}": "DELETE - Delete model"
        }
    }


@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy" if model is not None else "no_model",
        "model_loaded": model is not None,
        "current_model": current_model_name,
        "device": DEVICE,
        "models_directory": str(MODELS_DIR)
    }


# ============== Model Management API ==============

@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """Get list of all available models"""
    models = []
    
    for model_file in MODELS_DIR.glob("*.pt"):
        if model_file.name == "active_model.pt":
            continue
            
        size_mb = model_file.stat().st_size / (1024 * 1024)
        is_active = (current_model_name == model_file.name)
        
        models.append(ModelInfo(
            name=model_file.name,
            size_mb=round(size_mb, 2),
            is_active=is_active
        ))
    
    return models


@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    activate: bool = True
):
    """
    Upload new YOLOv5 model.
    
    - **file**: Model file (.pt)
    - **activate**: Activate model immediately after upload
    """
    logger.info(f"=== Starting model upload ===")
    logger.info(f"file.filename: {file.filename}")
    logger.info(f"file.content_type: {file.content_type}")
    logger.info(f"activate: {activate}")
    
    try:
        # Check filename
        if not file.filename:
            logger.error("Filename not specified")
            raise HTTPException(
                status_code=400,
                detail="Filename not specified"
            )
        
        if not file.filename.endswith(".pt"):
            logger.error(f"Invalid extension: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"File must have .pt extension (got: {file.filename})"
            )
        
        # Safe filename
        safe_filename = Path(file.filename).name
        model_path = MODELS_DIR / safe_filename
        
        # Read file
        contents = await file.read()
        logger.info(f"Bytes read: {len(contents)}")
        
        if len(contents) == 0:
            logger.error("File is empty")
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        logger.info(f"Saving to: {model_path}")
        
        with open(model_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"Model saved: {model_path}")
        
        # Validate model
        logger.info("Validating model...")
        if not validate_model(model_path):
            logger.error("Model validation failed")
            if model_path.exists():
                model_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="Invalid YOLOv5 model"
            )
        logger.info("Model is valid")
        
        result = {
            "success": True,
            "message": f"Model '{safe_filename}' uploaded successfully",
            "path": str(model_path),
            "size_mb": round(len(contents) / (1024 * 1024), 2)
        }
        
        # Activate model if needed
        if activate:
            shutil.copy(model_path, ACTIVE_MODEL_PATH)
            success = load_model(ACTIVE_MODEL_PATH)
            
            if success:
                result["activated"] = True
                result["message"] += " and activated"
            else:
                result["activated"] = False
                result["warning"] = "Model uploaded but could not be activated"
        else:
            result["activated"] = False
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model upload error: {e}")
        # Remove file on error
        if model_path.exists():
            model_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Model upload error: {str(e)}"
        )


@app.post("/models/activate/{model_name}")
async def activate_model(model_name: str):
    """
    Activate existing model.
    
    - **model_name**: Model filename (e.g., yolov5m.pt)
    """
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {[f.name for f in MODELS_DIR.glob('*.pt') if f.name != 'active_model.pt']}"
        )
    
    # Copy as active model
    shutil.copy(model_path, ACTIVE_MODEL_PATH)
    
    # Load model
    success = load_model(ACTIVE_MODEL_PATH)
    
    if success:
        return {
            "success": True,
            "message": f"Model '{model_name}' activated",
            "current_model": current_model_name
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Could not load model"
        )


@app.post("/models/upload/simple")
async def upload_model_simple(file: UploadFile = File(...)):
    """
    Simple model upload (no additional parameters).
    Model will be uploaded and activated immediately.
    
    - **file**: Model file (.pt)
    """
    # Check filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename not specified")
    
    if not file.filename.endswith(".pt"):
        raise HTTPException(
            status_code=400,
            detail=f"File must have .pt extension (got: {file.filename})"
        )
    
    safe_filename = Path(file.filename).name
    model_path = MODELS_DIR / safe_filename
    
    try:
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        logger.info(f"Received file: {safe_filename}, size: {len(contents)} bytes")
        
        with open(model_path, "wb") as f:
            f.write(contents)
        
        # Validate model
        if not validate_model(model_path):
            if model_path.exists():
                model_path.unlink()
            raise HTTPException(status_code=400, detail="Invalid YOLOv5 model")
        
        # Activate model
        shutil.copy(model_path, ACTIVE_MODEL_PATH)
        success = load_model(ACTIVE_MODEL_PATH)
        
        return {
            "success": True,
            "message": f"Model '{safe_filename}' uploaded and activated",
            "activated": success,
            "size_mb": round(len(contents) / (1024 * 1024), 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """
    Delete model.
    
    - **model_name**: Model filename
    """
    if model_name == "active_model.pt":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active model directly"
        )
    
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    # Check if model is active
    if current_model_name == model_name:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active model. Activate another model first."
        )
    
    model_path.unlink()
    
    return {
        "success": True,
        "message": f"Model '{model_name}' deleted"
    }


@app.get("/models/current")
async def get_current_model():
    """Get current active model information"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    size_mb = 0
    if ACTIVE_MODEL_PATH.exists():
        size_mb = ACTIVE_MODEL_PATH.stat().st_size / (1024 * 1024)
    
    return {
        "name": current_model_name,
        "device": DEVICE,
        "size_mb": round(size_mb, 2),
        "classes_count": len(model.names) if hasattr(model, 'names') else 0
    }


# ============== Detection API ==============

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.25,
    imgsz: ImageSize = ImageSize.SIZE_640
):
    """
    Object detection on image.
    
    - **file**: Image (JPEG, PNG)
    - **confidence_threshold**: Confidence threshold (0-1)
    - **imgsz**: Image size for inference (320, 416, 512, 640, 768, 1024, 1280)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Upload model via /models/upload"
        )
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        model.conf = confidence_threshold
        img_size = IMAGE_SIZES.get(imgsz.value, (640, 640))
        results = model(image, size=img_size[0])
        
        detections = []
        predictions = results.pandas().xyxy[0]
        
        for _, row in predictions.iterrows():
            detection = DetectionResult(
                class_name=row["name"],
                confidence=round(float(row["confidence"]), 4),
                bbox=[
                    round(float(row["xmin"]), 2),
                    round(float(row["ymin"]), 2),
                    round(float(row["xmax"]), 2),
                    round(float(row["ymax"]), 2)
                ]
            )
            detections.append(detection)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            total_objects=len(detections),
            model_name=current_model_name or "unknown"
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Image processing error: {str(e)}"
        )


@app.post("/detect/image")
async def detect_objects_with_image(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.25,
    imgsz: ImageSize = ImageSize.SIZE_640,
    line_thickness: int = 2
):
    """
    Object detection with image output containing bounding boxes.
    
    - **file**: Image (JPEG, PNG)
    - **confidence_threshold**: Confidence threshold (0-1)
    - **imgsz**: Image size for inference (320, 416, 512, 640, 768, 1024, 1280)
    - **line_thickness**: Bounding box thickness in pixels (1-10)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Limit line thickness
    line_thickness = max(1, min(10, line_thickness))
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        model.conf = confidence_threshold
        img_size = IMAGE_SIZES.get(imgsz.value, (640, 640))
        results = model(image, size=img_size[0])
        
        # Convert PIL Image to numpy array for drawing
        img_array = np.array(image)
        
        # Draw boxes with specified thickness
        img_with_boxes = draw_boxes(img_array, results, line_thickness)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(img_with_boxes)
        
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "inline; filename=detection_result.jpg",
                "X-Model-Name": current_model_name or "unknown"
            }
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Image processing error: {str(e)}"
        )


@app.post("/detect/batch")
async def detect_objects_batch(
    files: list[UploadFile] = File(...),
    confidence_threshold: float = 0.25,
    imgsz: ImageSize = ImageSize.SIZE_640
):
    """
    Batch image processing.
    
    - **files**: List of images
    - **confidence_threshold**: Confidence threshold (0-1)
    - **imgsz**: Image size for inference
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    img_size = IMAGE_SIZES.get(imgsz.value, (640, 640))
    results_list = []
    
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results_list.append({
                "filename": file.filename,
                "success": False,
                "error": "File is not an image"
            })
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            model.conf = confidence_threshold
            results = model(image, size=img_size[0])
            
            detections = []
            predictions = results.pandas().xyxy[0]
            
            for _, row in predictions.iterrows():
                detection = {
                    "class_name": row["name"],
                    "confidence": round(float(row["confidence"]), 4),
                    "bbox": [
                        round(float(row["xmin"]), 2),
                        round(float(row["ymin"]), 2),
                        round(float(row["xmax"]), 2),
                        round(float(row["ymax"]), 2)
                    ]
                }
                detections.append(detection)
            
            results_list.append({
                "filename": file.filename,
                "success": True,
                "detections": detections,
                "total_objects": len(detections)
            })
            
        except Exception as e:
            results_list.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "results": results_list,
        "model_name": current_model_name
    }


@app.get("/classes")
async def get_available_classes():
    """Get list of classes for current model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model": current_model_name,
        "classes": model.names,
        "total": len(model.names)
    }


@app.get("/sizes")
async def get_available_sizes():
    """Get list of available image sizes for inference"""
    return {
        "sizes": [
            {"value": "320", "dimensions": "320x320", "description": "Fast, low quality"},
            {"value": "416", "dimensions": "416x416", "description": "Fast"},
            {"value": "512", "dimensions": "512x512", "description": "Medium"},
            {"value": "640", "dimensions": "640x640", "description": "Standard (default)"},
            {"value": "768", "dimensions": "768x768", "description": "High quality"},
            {"value": "1024", "dimensions": "1024x1024", "description": "Very high quality"},
            {"value": "1280", "dimensions": "1280x1280", "description": "Maximum quality, slow"},
        ],
        "default": "640"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)