"""
DefectLens Evaluation Pipeline.

Usage:
------
python scripts/evaluate.py --config configs/mctf.yaml --category bottle

Overview:
---------
Evaluates a configured model architecture (Baseline ViT or MCTF-ViT) across
a complete MVTec category test split:
1. Iterates over test images with batch_size=1 to measure per-sample latency.
2. Extracts patch features and queries the category PatchCore memory bank.
3. Computes image-level AUROC and pixel-level AUROC against ground truth annotations.
4. Measures active token counts and token reduction percentage.
5. Saves structured evaluation metrics into `results/{model}_{category}_eval.json`.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json
import time

from src.models.vit import ViTFeatureExtractor
from src.models.anomaly_detector import PatchCoreAnomalyDetector
from src.data.dataset import MVTecDataset, get_transforms


def main():
    parser = argparse.ArgumentParser(description="Evaluate DefectLens model performance on MVTec")
    parser.add_argument('--config', type=str, default='configs/mctf.yaml', help='Path to configuration YAML')
    parser.add_argument('--category', type=str, default='bottle', help='Target MVTec category')
    parser.add_argument('--data_dir', type=str, default='data/mvtec', help='Path to dataset root')
    parser.add_argument('--memory_bank_dir', type=str, default='artifacts/memory_bank', help='Memory bank directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (1 for individual latency profiling)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset with image transformations and ground truth mask transforms
    transform, mask_transform = get_transforms(config.get('image_size', 224))
    dataset = MVTecDataset(root=args.data_dir, category=args.category, is_train=False, transform=transform, mask_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize ViT Feature Extractor
    model = ViTFeatureExtractor(
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        mctf_config=config.get('mctf')
    )
    model.to(device)
    model.eval()

    model_type = 'mctf' if config.get('mctf', {}).get('enabled', False) else 'baseline'
    mb_path = os.path.join(args.memory_bank_dir, args.category, model_type)
    
    # Load PatchCore memory bank
    detector = PatchCoreAnomalyDetector()
    detector.load(mb_path)

    layer_idx = config.get('layer_idx', -2)
    grid_size = config.get('image_size', 224) // model.patch_size

    img_scores = []
    img_labels = []
    pixel_scores = []
    pixel_labels = []
    latencies = []

    print(f"Evaluating {model_type} on {args.category}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            masks = batch['mask'].numpy()
            
            # Profile single-sample inference latency
            start_t = time.perf_counter()
            feat, mapping = model.extract_patch_features(images, layer_idx=layer_idx)
            anomaly_map, image_score = detector.predict(feat, mapping, patch_grid_size=grid_size)
            end_t = time.perf_counter()
            latencies.append((end_t - start_t) * 1000)
            
            img_scores.extend(image_score.cpu().numpy())
            img_labels.extend(labels)
            
            # Upsample coarse 14x14 anomaly map to original mask dimensions for pixel AUROC calculation
            B = images.shape[0]
            amap_upsampled = torch.nn.functional.interpolate(
                anomaly_map.unsqueeze(1), size=(masks.shape[2], masks.shape[3]), mode='bilinear', align_corners=False
            ).squeeze(1).cpu().numpy()
            
            pixel_scores.append(amap_upsampled.flatten())
            pixel_labels.append(masks.flatten())
            
    # Compute image-level AUROC
    if len(set(img_labels)) > 1:
        img_roc_auc = roc_auc_score(img_labels, img_scores)
    else:
        img_roc_auc = 0.0
        print("Warning: only one class present in test set; image AUROC undefined.")
    
    # Compute pixel-level AUROC
    pixel_scores = np.concatenate(pixel_scores)
    pixel_labels = np.concatenate(pixel_labels)
    pixel_labels = (pixel_labels > 0.5).astype(int)
    
    if len(np.unique(pixel_labels)) > 1:
        pixel_roc_auc = roc_auc_score(pixel_labels, pixel_scores)
    else:
        pixel_roc_auc = 0.0
        
    # Warmup latency adjustment: skip the first sample to eliminate initial load spikes
    timed = latencies[1:] if len(latencies) > 1 else latencies
    avg_latency = float(np.mean(timed)) if timed else 0.0

    # Calculate token reduction metrics
    orig_tokens = model.num_patches
    reduced_tokens = feat.shape[1]
    reduction_pct = (1.0 - reduced_tokens / orig_tokens) * 100
    
    results = {
        "model": model_type,
        "category": args.category,
        "image_auroc": img_roc_auc,
        "pixel_auroc": pixel_roc_auc,
        "avg_latency_ms": avg_latency,
        "token_reduction_pct": reduction_pct
    }
    
    print(json.dumps(results, indent=4))
    
    # Persist benchmark result JSON
    os.makedirs('results', exist_ok=True)
    res_path = os.path.join('results', f'{model_type}_{args.category}_eval.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
