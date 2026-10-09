"""
Heatmap and Visual Defect Localization Utilities.

Overview:
---------
Converts low-resolution anomaly score maps (14x14 patch grid) produced by PatchCore
into full-resolution (224x224), smoothed, color-mapped visualizations.

Key Operations:
1. Bicubic Interpolation: Upsamples the 14x14 coarse patch anomaly grid to the 224x224 input image resolution.
2. Gaussian Smoothing: Applies a 33x33 Gaussian filter to eliminate patch boundary artifacts and create continuous heat distributions.
3. Contrast-Preserving Normalization: Uses calibrated anomaly thresholds to avoid oversaturating nominal images while highlighting true defect regions.
4. Jet Colormap & Blending: Converts normalized anomaly magnitudes into RGB heatmaps (blue = normal, red = anomalous) and alpha-blends with the input image.
5. Defect Contour Extraction: Uses Otsu's binary thresholding and contour detection to localize and bound suspicious defect regions.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os


def generate_heatmap(
    anomaly_map: torch.Tensor,
    original_image: np.ndarray,
    alpha: float = 0.5,
    threshold: float = None
) -> dict:
    """
    Transforms a coarse 2D patch anomaly map into publication-quality inspection visuals.
    
    Args:
        anomaly_map (torch.Tensor): Coarse anomaly map [14, 14] or [1, 1, 14, 14].
        original_image (np.ndarray): Original RGB image array [H, W, 3] with values in [0, 255].
        alpha (float): Blending factor for the heatmap overlay (0.0 = original only, 1.0 = heatmap only).
        threshold (float, optional): Calibrated category threshold used to normalize anomaly values
            without oversaturating nominal samples.
            
    Returns:
        dict:
            - 'heatmap': RGB color-mapped anomaly visualization.
            - 'overlay': Blended visualization (original image + heatmap).
            - 'region': Original image with localized defect contour boundaries in red.
            - 'raw_map': Continuous float32 smoothed anomaly map.
    """
    # Ensure 4D tensor shape [1, 1, H_patch, W_patch] for interpolation
    if len(anomaly_map.shape) == 2:
        anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)
    elif len(anomaly_map.shape) == 3:
        anomaly_map = anomaly_map.unsqueeze(0)
        
    H, W = original_image.shape[:2]
    
    # 1. Upsample 14x14 coarse patch anomaly map to original image resolution (e.g., 224x224)
    anomaly_map_resized = F.interpolate(anomaly_map, size=(H, W), mode='bicubic', align_corners=False)
    anomaly_map_resized = anomaly_map_resized.squeeze().cpu().numpy()
    
    # 2. Apply Gaussian smoothing to eliminate patch boundary discontinuities
    anomaly_map_smoothed = cv2.GaussianBlur(anomaly_map_resized, (33, 33), 0)
    
    # 3. Dynamic Normalization into [0, 1] range
    min_val, max_val = anomaly_map_smoothed.min(), anomaly_map_smoothed.max()
    if threshold is not None and threshold > 0:
        # Anchor the ceiling to the higher of max anomaly score or classification threshold
        norm_max = max(max_val, threshold)
        if norm_max > min_val:
            norm_map = (anomaly_map_smoothed - min_val) / (norm_max - min_val)
            norm_map = np.clip(norm_map, 0, 1)
        else:
            norm_map = np.zeros_like(anomaly_map_smoothed)
    else:
        if max_val > min_val:
            norm_map = (anomaly_map_smoothed - min_val) / (max_val - min_val)
        else:
            norm_map = np.zeros_like(anomaly_map_smoothed)
        
    # Scale to uint8 [0, 255]
    heatmap_255 = (norm_map * 255).astype(np.uint8)
    
    # 4. Color-map conversion (COLORMAP_JET: Blue = 0, Red = 255)
    colormap = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    # 5. Alpha blending with original image
    overlay = cv2.addWeighted(original_image, 1 - alpha, colormap, alpha, 0)
    
    # 6. Automatic suspicious defect segmentation via Otsu thresholding
    _, thresh = cv2.threshold(heatmap_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contour outlines in red around high-anomaly regions
    region_img = original_image.copy()
    cv2.drawContours(region_img, contours, -1, (255, 0, 0), 2)
    
    return {
        'heatmap': colormap,
        'overlay': overlay,
        'region': region_img,
        'raw_map': anomaly_map_smoothed
    }


def save_visualizations(results: dict, save_dir: str, prefix: str = ""):
    """Saves generated visualization maps to disk as PNG files."""
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(results['heatmap']).save(os.path.join(save_dir, f"{prefix}heatmap.png"))
    Image.fromarray(results['overlay']).save(os.path.join(save_dir, f"{prefix}overlay.png"))
    Image.fromarray(results['region']).save(os.path.join(save_dir, f"{prefix}region.png"))
