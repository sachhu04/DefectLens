import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
import sys
import time
import torch
import numpy as np

print("=" * 70)
print(" DEFECTLENS: END-TO-END MODULE INTEGRATION & SYSTEM TEST SUITE ")
print("=" * 70)

# 1. Dataset & Preprocessing
print("\n[TEST 1/5] Testing Dataset Loader & Torchvision Transforms...")
from src.data.dataset import MVTecDataset, get_transforms
t, mt = get_transforms(224)
dataset = MVTecDataset(root='data/mvtec', category='bottle', is_train=False, transform=t, mask_transform=mt)
assert len(dataset) > 0, "Dataset should have samples"
sample = dataset[0]
print(f" -> Dataset loaded successfully: {len(dataset)} test samples available.")
print(f" -> Sample tensor shape: {sample['image'].shape}, label: {sample['label']}")
print(" [PASSED] Module 1: Dataset & Preprocessing verified.")

# 2. ViT Backbone & MCTF Module
print("\n[TEST 2/5] Testing ViT Backbone with Multi-Criteria Token Fusion (MCTF)...")
from src.models.vit import ViTFeatureExtractor
mctf_config = {
    'enabled': True,
    'reduction_ratio': 0.2,
    'temperature_sim': 1.0,
    'temperature_info': 1.0,
    'temperature_size': 1.0,
    'use_sim': True,
    'use_info': True,
    'use_size': True
}
model_mctf = ViTFeatureExtractor(model_name='deit_small_patch16_224', pretrained=True, mctf_config=mctf_config)
model_mctf.eval()

img = sample['image'].unsqueeze(0)
with torch.no_grad():
    feat, mapping = model_mctf.extract_patch_features(img, layer_idx=-2)

orig_tokens = model_mctf.num_patches
fused_tokens = feat.shape[1]
reduction = (1.0 - fused_tokens / orig_tokens) * 100
print(f" -> ViT + MCTF forward pass complete.")
print(f" -> Original patch tokens: {orig_tokens}, Fused tokens: {fused_tokens}")
print(f" -> Dynamic Token Reduction: {reduction:.1f}%")
assert fused_tokens < orig_tokens, "MCTF should fuse tokens"
print(" [PASSED] Module 2: ViT Backbone & MCTF Token Fusion verified.")

# 3. FAISS Memory Bank & Anomaly Detector
print("\n[TEST 3/5] Testing FAISS Memory Bank Matching & Scoring...")
from src.models.anomaly_detector import PatchCoreAnomalyDetector, HAS_FAISS
print(f" -> FAISS acceleration engine active: {HAS_FAISS}")
detector = PatchCoreAnomalyDetector()
detector.load('artifacts/memory_bank/bottle/mctf')
assert detector.memory_bank is not None, "Memory bank must be loaded"

with torch.no_grad():
    anomaly_map, image_score = detector.predict(feat, mapping, patch_grid_size=14)
print(f" -> Nearest-neighbor search executed on memory bank (ntotal={detector.memory_bank.shape[0]}).")
print(f" -> Anomaly Map grid: {anomaly_map.shape}, Image-level score: {image_score.item():.4f}")
print(" [PASSED] Module 3: Memory Bank & Anomaly Scoring verified.")

# 4. Heatmap Generation & Visual Explainability
print("\n[TEST 4/5] Testing Heatmap Spatial Reconstruction & Visualization...")
from src.visualization.heatmap import generate_heatmap
img_np = np.zeros((224, 224, 3), dtype=np.uint8)
vis = generate_heatmap(anomaly_map[0], img_np, threshold=0.69)
assert 'heatmap' in vis and 'overlay' in vis and 'region' in vis
print(" -> Heatmap, overlay, and suspicious region contours synthesized.")
print(" [PASSED] Module 4: Heatmap Generation & Visual Output verified.")

# 5. FastAPI REST API Integration
print("\n[TEST 5/5] Testing FastAPI REST Backend Endpoints...")
from fastapi.testclient import TestClient
from backend.app.main import app
client = TestClient(app)

res_health = client.get("/health")
assert res_health.status_code == 200
print(f" -> GET /health response: {res_health.json()}")

res_models = client.get("/model-info")
assert res_models.status_code == 200
print(f" -> GET /model-info response: {res_models.json()['models']}")

res_cats = client.get("/categories")
assert res_cats.status_code == 200
print(f" -> GET /categories response: {res_cats.json()['categories']}")

# Test /predict endpoint with test image
import io
from fastapi import UploadFile
from backend.app.main import predict
test_img_path = 'data/mvtec/bottle/test/broken_small/000.png'
with open(test_img_path, 'rb') as f:
    uf = UploadFile(filename='000.png', file=io.BytesIO(f.read()))
    res_pred = predict(file=uf, category='bottle', model_type='mctf')
assert res_pred.status_code == 200
import json
pred_data = json.loads(res_pred.body)
print(f" -> POST /predict: prediction='{pred_data['prediction']}', score={pred_data['anomaly_score']:.4f}, latency={pred_data['inference_ms']:.1f}ms, token_reduction={pred_data['token_reduction_percent']:.1f}%")
print(" [PASSED] Module 5: FastAPI REST Backend Service verified.")

print("\n" + "=" * 70)
print(" ALL 5 MODULES SUCCESSFULLY INTEGRATED AND VERIFIED (100% PASS) ")
print("=" * 70)
