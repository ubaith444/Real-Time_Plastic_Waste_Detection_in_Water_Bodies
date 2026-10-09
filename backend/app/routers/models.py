from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.config import MODELS_DIR
from backend.app.cv.camera_stream import stream_manager

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("")
def list_models():
    """
    Lists all available and imported YOLO/ONNX model weight files.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    
    # Check default model in detector
    active_model_name = getattr(stream_manager.detector, "current_model_name", "yolov8n.pt")

    # List all files in MODELS_DIR
    for p in MODELS_DIR.glob("*"):
        if p.suffix.lower() in [".pt", ".onnx"]:
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            is_active = (p.name == active_model_name) or (p.stem == Path(active_model_name).stem)
            models.append({
                "filename": p.name,
                "format": "ONNX" if p.suffix.lower() == ".onnx" else "PyTorch",
                "size_mb": size_mb,
                "modified_at": mtime,
                "is_active": is_active,
                "path": str(p)
            })

    # Ensure baseline yolov8n.pt is listed even if relying on torch cache
    has_default_pt = any(m["filename"] == "yolov8n.pt" for m in models)
    if not has_default_pt:
        models.insert(0, {
            "filename": "yolov8n.pt",
            "format": "PyTorch",
            "size_mb": 6.2,
            "modified_at": "Baseline",
            "is_active": active_model_name == "yolov8n.pt",
            "path": "ultralytics/yolov8n.pt"
        })

    return models

@router.post("/import")
async def import_model(file: UploadFile = File(...)):
    """
    Imports a custom trained model weight file (.pt or .onnx) into the platform.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
        
    ext = Path(filename).suffix.lower()
    if ext not in [".pt", ".onnx"]:
        raise HTTPException(status_code=400, detail="Unsupported model format. Only .pt and .onnx model weights are supported.")

    dest_path = MODELS_DIR / filename
    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        
        size_mb = round(len(content) / (1024 * 1024), 2)
        return {
            "status": "success",
            "message": f"Successfully imported model weights: {filename}",
            "filename": filename,
            "size_mb": size_mb,
            "format": "ONNX" if ext == ".onnx" else "PyTorch"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import model: {str(e)}")

@router.post("/{model_name}/activate")
def activate_model(model_name: str):
    """
    Dynamically switches the active detector model weights to the specified model.
    """
    target_path = MODELS_DIR / model_name
    if not target_path.exists() and model_name != "yolov8n.pt":
        raise HTTPException(status_code=404, detail="Specified model file not found.")

    try:
        from ultralytics import YOLO
        if target_path.exists():
            stream_manager.detector.yolo_model = YOLO(str(target_path))
            stream_manager.detector.model_path = target_path
            stream_manager.detector.current_model_name = model_name
        else:
            stream_manager.detector.yolo_model = YOLO("yolov8n.pt")
            stream_manager.detector.current_model_name = "yolov8n.pt"

        return {
            "status": "success",
            "active_model": model_name,
            "message": f"Inference engine switched to {model_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")
