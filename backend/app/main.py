from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import yaml
import tempfile
import base64
from PIL import Image
import torch
import numpy as np

import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.vit import ViTFeatureExtractor
from src.models.anomaly_detector import PatchCoreAnomalyDetector
from src.data.dataset import get_transforms
from src.visualization.heatmap import generate_heatmap

app = FastAPI(title="Efficient ViT Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache to avoid reloading
MODELS = {}
DETECTORS = {}
CONFIGS = {}

def load_config(model_type):
    if model_type not in CONFIGS:
        with open(f"configs/{model_type}.yaml", 'r') as f:
            CONFIGS[model_type] = yaml.safe_load(f)
    return CONFIGS[model_type]

def get_model(model_type, device):
    if model_type not in MODELS:
        config = load_config(model_type)
        model = ViTFeatureExtractor(
            model_name=config['model_name'],
            pretrained=config['pretrained'],
            mctf_config=config.get('mctf')
        )
        model.to(device)
        model.eval()
        MODELS[model_type] = model
    return MODELS[model_type]

def get_detector(model_type, category):
    key = f"{model_type}_{category}"
    if key not in DETECTORS:
        config = load_config(model_type)
        detector = PatchCoreAnomalyDetector(
            coreset_sampling_ratio=config['anomaly_detector'].get('coreset_sampling_ratio', 0.1),
            num_neighbors=config['anomaly_detector'].get('num_neighbors', 9)
        )
        mb_path = os.path.join('artifacts/memory_bank', category, model_type)
        if not os.path.exists(mb_path):
            raise HTTPException(status_code=400, detail=f"Memory bank not found for {category} with model {model_type}. Please build it first.")
        detector.load(mb_path)
        DETECTORS[key] = detector
    return DETECTORS[key]

def image_to_base64(img_np):
    import cv2
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.get("/categories")
def get_categories():
    base_dir = "data/mvtec"
    if not os.path.exists(base_dir):
        return {"categories": []}
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return {"categories": sorted(categories)}

@app.get("/model-info")
def get_model_info():
    return {
        "models": ["baseline", "mctf"],
        "description": "Baseline is standard ViT. MCTF applies Multi-Criteria Token Fusion."
    }

import time

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    category: str = Form("bottle"),
    model_type: str = Form("mctf")
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = get_model(model_type, device)
        detector = get_detector(model_type, category)
        config = load_config(model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    try:
        img_pil = Image.open(file.file).convert('RGB')
        img_size = config.get('image_size', 224)
        transform, _ = get_transforms(img_size)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        img_np = np.array(img_pil.resize((img_size, img_size), Image.Resampling.BICUBIC))
        
        layer_idx = config.get('layer_idx', -2)
        grid_size = img_size // 16
        
        start_time = time.perf_counter()
        with torch.no_grad():
            feat, mapping = model.extract_patch_features(img_tensor, layer_idx=layer_idx)
            anomaly_map, image_score = detector.predict(feat, mapping, patch_grid_size=grid_size)
        end_time = time.perf_counter()
        
        inference_ms = (end_time - start_time) * 1000
        image_score_val = float(image_score.item())
        prediction_label = "anomalous" if image_score_val > 25.0 else "normal" # Adjusted for real dataset
        
        orig_tokens = model.num_patches
        reduced_tokens = feat.shape[1]
        reduction_pct = (1.0 - reduced_tokens / orig_tokens) * 100
        
        # Heatmap
        vis = generate_heatmap(anomaly_map[0], img_np)
        
        return JSONResponse({
            "prediction": prediction_label,
            "anomaly_score": image_score_val,
            "inference_ms": inference_ms,
            "original_tokens": orig_tokens,
            "reduced_tokens": reduced_tokens,
            "token_reduction_percent": float(reduction_pct),
            "images": {
                "original": image_to_base64(img_np),
                "heatmap": image_to_base64(vis['heatmap']),
                "overlay": image_to_base64(vis['overlay']),
                "region": image_to_base64(vis['region'])
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
