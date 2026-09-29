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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mctf.yaml')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--data_dir', type=str, default='data/mvtec')
    parser.add_argument('--memory_bank_dir', type=str, default='artifacts/memory_bank')
    parser.add_argument('--batch_size', type=int, default=1) # batch=1 for latency timing
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform, mask_transform = get_transforms(config.get('image_size', 224))
    dataset = MVTecDataset(root=args.data_dir, category=args.category, is_train=False, transform=transform, mask_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ViTFeatureExtractor(
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        mctf_config=config.get('mctf')
    )
    model.to(device)
    model.eval()

    model_type = 'mctf' if config.get('mctf', {}).get('enabled', False) else 'baseline'
    mb_path = os.path.join(args.memory_bank_dir, args.category, model_type)
    
    detector = PatchCoreAnomalyDetector()
    detector.load(mb_path)

    layer_idx = config.get('layer_idx', -2)
    grid_size = config.get('image_size', 224) // model.patch_size

    img_scores = []
    img_labels = []
    
    # Optional pixel-level eval
    pixel_scores = []
    pixel_labels = []

    latencies = []

    print(f"Evaluating {model_type} on {args.category}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            masks = batch['mask'].numpy()
            
            start_t = time.perf_counter()
            feat, mapping = model.extract_patch_features(images, layer_idx=layer_idx)
            anomaly_map, image_score = detector.predict(feat, mapping, patch_grid_size=grid_size)
            end_t = time.perf_counter()
            latencies.append((end_t - start_t) * 1000)
            
            img_scores.extend(image_score.cpu().numpy())
            img_labels.extend(labels)
            
            # Simple upsample of anomaly map for pixel auroc
            # In a real scenario, use F.interpolate on anomaly_map directly
            B = images.shape[0]
            amap_upsampled = torch.nn.functional.interpolate(
                anomaly_map.unsqueeze(1), size=(masks.shape[2], masks.shape[3]), mode='bilinear', align_corners=False
            ).squeeze(1).cpu().numpy()
            
            pixel_scores.append(amap_upsampled.flatten())
            pixel_labels.append(masks.flatten())
            
    if len(set(img_labels)) > 1:
        img_roc_auc = roc_auc_score(img_labels, img_scores)
    else:
        img_roc_auc = 0.0
        print("Warning: only one class present in test set; image AUROC undefined.")
    
    pixel_scores = np.concatenate(pixel_scores)
    pixel_labels = np.concatenate(pixel_labels)
    # Threshold mask to 0/1 (some might be continuous if resized)
    pixel_labels = (pixel_labels > 0.5).astype(int)
    
    # Avoid calculating pixel AUROC if memory is too high or all labels are 0
    if len(np.unique(pixel_labels)) > 1:
        pixel_roc_auc = roc_auc_score(pixel_labels, pixel_scores)
    else:
        pixel_roc_auc = 0.0
        
    # Skip first iteration for warmup; fall back to all samples if only one
    timed = latencies[1:] if len(latencies) > 1 else latencies
    avg_latency = float(np.mean(timed)) if timed else 0.0

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
    
    os.makedirs('results', exist_ok=True)
    res_path = os.path.join('results', f'{model_type}_{args.category}_eval.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
