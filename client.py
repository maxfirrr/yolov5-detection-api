"""
Example client for Object Detection API (with model management)
"""

import requests
import sys
from pathlib import Path


API_URL = "http://localhost:8000"


# ============== Model Management ==============

def list_models() -> list:
    """Get list of all models"""
    response = requests.get(f"{API_URL}/models")
    response.raise_for_status()
    return response.json()


def get_current_model() -> dict:
    """Get current model information"""
    response = requests.get(f"{API_URL}/models/current")
    response.raise_for_status()
    return response.json()


def upload_model(model_path: str, activate: bool = True) -> dict:
    """Upload new model"""
    with open(model_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/models/upload",
            files={"file": f},
            params={"activate": str(activate).lower()}
        )
    response.raise_for_status()
    return response.json()


def activate_model(model_name: str) -> dict:
    """Activate model by name"""
    response = requests.post(f"{API_URL}/models/activate/{model_name}")
    response.raise_for_status()
    return response.json()


def delete_model(model_name: str) -> dict:
    """Delete model"""
    response = requests.delete(f"{API_URL}/models/{model_name}")
    response.raise_for_status()
    return response.json()


# ============== Detection ==============

def detect_objects(image_path: str, confidence: float = 0.25, imgsz: str = "640") -> dict:
    """Detect objects on image"""
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/detect",
            files={"file": f},
            params={"confidence_threshold": confidence, "imgsz": imgsz}
        )
    response.raise_for_status()
    return response.json()


def detect_and_save_image(
    image_path: str, 
    output_path: str, 
    confidence: float = 0.25,
    imgsz: str = "640",
    line_thickness: int = 2
):
    """Detect objects and save image with bounding boxes"""
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/detect/image",
            files={"file": f},
            params={
                "confidence_threshold": confidence,
                "imgsz": imgsz,
                "line_thickness": line_thickness
            }
        )
    response.raise_for_status()
    
    with open(output_path, "wb") as out:
        out.write(response.content)
    print(f"Result saved to {output_path}")


def get_classes() -> dict:
    """Get list of available classes"""
    response = requests.get(f"{API_URL}/classes")
    response.raise_for_status()
    return response.json()


def get_sizes() -> dict:
    """Get list of available image sizes"""
    response = requests.get(f"{API_URL}/sizes")
    response.raise_for_status()
    return response.json()


def health_check() -> dict:
    """Check service health"""
    response = requests.get(f"{API_URL}/health")
    response.raise_for_status()
    return response.json()


def print_help():
    """Print help"""
    print("""
Object Detection API Client

Usage:
    python client.py detect <image_path> [confidence] [imgsz]  - Detect objects
    python client.py image <image_path> [confidence] [imgsz] [line_thickness] - Get image with boxes
    python client.py models                                    - List models
    python client.py current                                   - Current model
    python client.py upload <model_path> [activate]            - Upload model
    python client.py activate <model_name>                     - Activate model
    python client.py delete <model_name>                       - Delete model
    python client.py classes                                   - List classes
    python client.py sizes                                     - List available sizes
    python client.py health                                    - Service health

Examples:
    python client.py detect photo.jpg 0.5 640
    python client.py image photo.jpg 0.5 1024 3
    python client.py upload yolov5m.pt true
    python client.py activate yolov5m.pt
""")


def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "health":
            print(health_check())
        
        elif command == "models":
            models = list_models()
            if not models:
                print("No models loaded")
            else:
                print("Available models:")
                for m in models:
                    active = " [ACTIVE]" if m["is_active"] else ""
                    print(f"  - {m['name']} ({m['size_mb']:.1f} MB){active}")
        
        elif command == "current":
            model = get_current_model()
            print(f"Current model: {model['name']}")
            print(f"  Device: {model['device']}")
            print(f"  Size: {model['size_mb']:.1f} MB")
            print(f"  Classes: {model['classes_count']}")
        
        elif command == "upload":
            if len(sys.argv) < 3:
                print("Specify model path")
                return
            model_path = sys.argv[2]
            activate = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True
            result = upload_model(model_path, activate)
            print(result["message"])
        
        elif command == "activate":
            if len(sys.argv) < 3:
                print("Specify model name")
                return
            result = activate_model(sys.argv[2])
            print(result["message"])
        
        elif command == "delete":
            if len(sys.argv) < 3:
                print("Specify model name")
                return
            result = delete_model(sys.argv[2])
            print(result["message"])
        
        elif command == "classes":
            data = get_classes()
            print(f"Model: {data['model']}")
            print(f"Classes: {data['total']}")
            print("Classes:", ", ".join(data['classes'].values()))
        
        elif command == "sizes":
            data = get_sizes()
            print("Available sizes:")
            for size in data['sizes']:
                default = " (default)" if size['value'] == data['default'] else ""
                print(f"  {size['value']}: {size['dimensions']} - {size['description']}{default}")
        
        elif command == "detect":
            if len(sys.argv) < 3:
                print("Specify image path")
                return
            
            image_path = sys.argv[2]
            confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
            imgsz = sys.argv[4] if len(sys.argv) > 4 else "640"
            
            if not Path(image_path).exists():
                print(f"File not found: {image_path}")
                return
            
            print(f"Analyzing: {image_path} (threshold: {confidence}, size: {imgsz})")
            result = detect_objects(image_path, confidence, imgsz)
            
            print(f"Model: {result['model_name']}")
            print(f"Objects found: {result['total_objects']}")
            
            for det in result['detections']:
                print(f"  {det['class_name']}: {det['confidence']:.2%}")
        
        elif command == "image":
            if len(sys.argv) < 3:
                print("Specify image path")
                return
            
            image_path = sys.argv[2]
            confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
            imgsz = sys.argv[4] if len(sys.argv) > 4 else "640"
            line_thickness = int(sys.argv[5]) if len(sys.argv) > 5 else 2
            
            if not Path(image_path).exists():
                print(f"File not found: {image_path}")
                return
            
            output_path = Path(image_path).stem + "_detected.jpg"
            detect_and_save_image(image_path, output_path, confidence, imgsz, line_thickness)
        
        else:
            print_help()
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server")
        print("Make sure server is running at http://localhost:8000")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        try:
            print(e.response.json().get("detail", "Unknown error"))
        except:
            print(e.response.text)


if __name__ == "__main__":
    main()