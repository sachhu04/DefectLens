import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def generate_heatmap(anomaly_map: torch.Tensor, original_image: np.ndarray, alpha: float = 0.5) -> dict:
    """
    Generates a heatmap and overlays it on the original image.
    
    anomaly_map: [H_patch, W_patch] or [1, 1, H_patch, W_patch] torch tensor.
    original_image: [H, W, 3] numpy array (RGB, 0-255).
    alpha: blending factor for overlay.
    
    Returns:
        dict with 'heatmap', 'overlay', 'thresholded'
    """
    if len(anomaly_map.shape) == 2:
        anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)
    elif len(anomaly_map.shape) == 3:
        anomaly_map = anomaly_map.unsqueeze(0)
        
    H, W = original_image.shape[:2]
    
    # Upsample anomaly map to original image size
    anomaly_map_resized = F.interpolate(anomaly_map, size=(H, W), mode='bicubic', align_corners=False)
    anomaly_map_resized = anomaly_map_resized.squeeze().cpu().numpy()
    
    # Apply Gaussian blur for smoothness
    anomaly_map_smoothed = cv2.GaussianBlur(anomaly_map_resized, (33, 33), 0)
    
    # Normalize to 0-255
    min_val, max_val = anomaly_map_smoothed.min(), anomaly_map_smoothed.max()
    if max_val > min_val:
        norm_map = (anomaly_map_smoothed - min_val) / (max_val - min_val)
    else:
        norm_map = np.zeros_like(anomaly_map_smoothed)
        
    heatmap_255 = (norm_map * 255).astype(np.uint8)
    
    # Apply colormap
    colormap = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(original_image, 1 - alpha, colormap, alpha, 0)
    
    # Threshold for suspicious region
    # Simple Otsu's thresholding or a fixed threshold on normalized map
    _, thresh = cv2.threshold(heatmap_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    region_img = original_image.copy()
    cv2.drawContours(region_img, contours, -1, (255, 0, 0), 2)
    
    return {
        'heatmap': colormap,
        'overlay': overlay,
        'region': region_img,
        'raw_map': anomaly_map_smoothed
    }

def save_visualizations(results: dict, save_dir: str, prefix: str = ""):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    Image.fromarray(results['heatmap']).save(os.path.join(save_dir, f"{prefix}heatmap.png"))
    Image.fromarray(results['overlay']).save(os.path.join(save_dir, f"{prefix}overlay.png"))
    Image.fromarray(results['region']).save(os.path.join(save_dir, f"{prefix}region.png"))
