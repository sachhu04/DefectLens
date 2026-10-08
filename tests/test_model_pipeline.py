import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pytest

from src.models.vit import ViTFeatureExtractor
from src.models.anomaly_detector import PatchCoreAnomalyDetector
from src.visualization.heatmap import generate_heatmap
from src.data.dataset import MVTecDataset, get_transforms

def test_dataset_loader():
    t, mt = get_transforms(224)
    dataset = MVTecDataset(root='data/mvtec', category='bottle', is_train=False, transform=t, mask_transform=mt)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample['image'].shape == (3, 224, 224)
    assert 'label' in sample
    assert 'mask' in sample

def test_vit_mctf_forward_pass():
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
    model = ViTFeatureExtractor(model_name='deit_small_patch16_224', pretrained=True, mctf_config=mctf_config)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feat, mapping = model.extract_patch_features(dummy_input, layer_idx=-2)
    
    orig_tokens = model.num_patches
    fused_tokens = feat.shape[1]
    assert fused_tokens < orig_tokens
    assert mapping.shape[0] == 1
    assert mapping.shape[1] == orig_tokens

def test_memory_bank_and_scoring():
    detector = PatchCoreAnomalyDetector()
    mb_path = 'artifacts/memory_bank/bottle/mctf'
    detector.load(mb_path)
    assert detector.memory_bank is not None
    assert detector.memory_bank.shape[1] == 384

    # Test prediction with dummy features and mapping
    dummy_feat = torch.randn(1, 23, 384)
    dummy_mapping = torch.randint(0, 23, (1, 196))
    with torch.no_grad():
        anomaly_map, image_score = detector.predict(dummy_feat, dummy_mapping, patch_grid_size=14)
    assert anomaly_map.shape == (1, 14, 14)
    assert image_score.item() >= 0.0

def test_heatmap_generation():
    dummy_anomaly_map = torch.rand(14, 14)
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    vis = generate_heatmap(dummy_anomaly_map, dummy_image, threshold=0.7)
    assert 'heatmap' in vis
    assert 'overlay' in vis
    assert 'region' in vis
    assert vis['heatmap'].shape == (224, 224, 3)
