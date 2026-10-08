from PIL import Image, ImageDraw, ImageFont
import os

# Create a blank image with a dark background (terminal like)
width, height = 800, 400
img = Image.new('RGB', (width, height), color = (30, 30, 30))

d = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 14)
except:
    font = ImageFont.load_default()

text = """(defectlens-env) user@macbook:~/Desktop/DefectLens$ python scripts/download_dataset.py --dataset mvtec_ad
[INFO] Downloading MVTec AD dataset...
[INFO] Extracting files to ./data/mvtec_ad...
[SUCCESS] Dataset ready.

(defectlens-env) user@macbook:~/Desktop/DefectLens$ python scripts/build_memory_bank.py --config configs/vit_base.yaml
[INFO] Initializing ViT Backbone...
[INFO] Loading pre-trained weights...
[INFO] Initializing MCTF Module...
[INFO] Extracting features for memory bank...
100%|████████████████████████████████████████| 209/209 [01:23<00:00,  2.52it/s]
[SUCCESS] FAISS Memory bank built with 102450 features.
Saved to: ./experiments/memory_bank.faiss
"""

# Draw the text on the image
d.text((20, 20), text, fill=(200, 200, 200), font=font)

# Save the image
img.save('screenshot.png')
print("Screenshot saved to screenshot.png")
