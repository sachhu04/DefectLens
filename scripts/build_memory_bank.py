import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vit import ViTFeatureExtractor
from src.models.anomaly_detector import PatchCoreAnomalyDetector
from src.data.dataset import MVTecDataset, get_transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--data_dir', type=str, default='data/mvtec')
    parser.add_argument('--save_dir', type=str, default='artifacts/memory_bank')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform, _ = get_transforms(config.get('image_size', 224))
    
    dataset = MVTecDataset(root=args.data_dir, category=args.category, is_train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ViTFeatureExtractor(
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        mctf_config=config.get('mctf')
    )
    model.to(device)
    model.eval()

    detector = PatchCoreAnomalyDetector(
        coreset_sampling_ratio=config['anomaly_detector'].get('coreset_sampling_ratio', 0.1),
        num_neighbors=config['anomaly_detector'].get('num_neighbors', 9)
    )

    layer_idx = config.get('layer_idx', -2)
    features_list = []

    print("Extracting features...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            # We just need the features, we don't care about mapping for the memory bank construction
            feat, mapping = model.extract_patch_features(images, layer_idx=layer_idx)
            # feat shape: [B, N, C]
            B, N, C = feat.shape
            feat = feat.reshape(B * N, C)
            features_list.append(feat.cpu())

    print("Building memory bank (coreset sampling)...")
    detector.fit(features_list)

    # Save
    model_type = 'mctf' if config.get('mctf', {}).get('enabled', False) else 'baseline'
    save_path = os.path.join(args.save_dir, args.category, model_type)
    detector.save(save_path)
    print(f"Memory bank saved to {save_path}")

if __name__ == "__main__":
    main()
