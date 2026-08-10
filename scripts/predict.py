import os
import argparse
import yaml
import torch
import time
import json
from PIL import Image
import numpy as np

from src.models.vit import ViTFeatureExtractor
from src.models.anomaly_detector import PatchCoreAnomalyDetector
from src.data.dataset import get_transforms
from src.visualization.heatmap import generate_heatmap, save_visualizations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--model', type=str, choices=['baseline', 'mctf'], default='mctf')
    parser.add_argument('--memory_bank_dir', type=str, default='artifacts/memory_bank')
    parser.add_argument('--output_dir', type=str, default='outputs/prediction')
    args = parser.parse_args()

    config_path = f'configs/{args.model}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = ViTFeatureExtractor(
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        mctf_config=config.get('mctf')
    )
    model.to(device)
    model.eval()

    # Load detector memory bank
    detector = PatchCoreAnomalyDetector(
        coreset_sampling_ratio=config['anomaly_detector'].get('coreset_sampling_ratio', 0.1),
        num_neighbors=config['anomaly_detector'].get('num_neighbors', 9)
    )
    mb_path = os.path.join(args.memory_bank_dir, args.category, args.model)
    detector.load(mb_path)

    # Process image
    transform, _ = get_transforms(config.get('image_size', 224))
    img_pil = Image.open(args.image).convert('RGB')
    img_np = np.array(img_pil.resize((config.get('image_size', 224), config.get('image_size', 224)), Image.Resampling.BICUBIC))
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    layer_idx = config.get('layer_idx', -2)
    grid_size = config.get('image_size', 224) // 16 # Assuming patch16

    # Inference timing
    start_time = time.perf_counter()
    
    with torch.no_grad():
        feat, mapping = model.extract_patch_features(img_tensor, layer_idx=layer_idx)
        anomaly_map, image_score = detector.predict(feat, mapping, patch_grid_size=grid_size)
        
    end_time = time.perf_counter()
    inference_ms = (end_time - start_time) * 1000

    image_score_val = image_score.item()
    prediction_label = "ANOMALOUS" if image_score_val > 25.0 else "NORMAL" # Adjusted for real dataset

    # Calculate token statistics
    orig_tokens = model.num_patches
    # feat shape [1, N_reduced, C]
    reduced_tokens = feat.shape[1]
    reduction_pct = (1.0 - reduced_tokens / orig_tokens) * 100

    print(f"Prediction: {prediction_label}")
    print(f"Anomaly Score: {image_score_val:.4f}")
    print(f"Inference Time: {inference_ms:.2f} ms")
    print(f"Original Tokens: {orig_tokens}")
    print(f"Reduced Tokens: {reduced_tokens}")
    print(f"Token Reduction: {reduction_pct:.1f}%")

    # Generate heatmaps
    vis_results = generate_heatmap(anomaly_map[0], img_np)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_visualizations(vis_results, args.output_dir)

    result_json = {
        "prediction": prediction_label.lower(),
        "anomaly_score": image_score_val,
        "inference_ms": inference_ms,
        "original_tokens": orig_tokens,
        "reduced_tokens": reduced_tokens,
        "token_reduction_percent": reduction_pct
    }

    with open(os.path.join(args.output_dir, 'result.json'), 'w') as f:
        json.dump(result_json, f, indent=4)

if __name__ == "__main__":
    main()
