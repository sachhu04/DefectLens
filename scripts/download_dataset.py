import os
import sys
import tarfile
import urllib.request
import argparse

# Some community mirrors for MVTec AD categories (these can sometimes expire)
MIRRORS = {
    'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz'
}

# Hugging Face mirror that keeps the original MVTec AD folder layout
# (train/good, test/<defect>, ground_truth/<defect>) for every category.
HF_REPO = "foersben/mvtec-ad"

MANUAL_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"


def download_and_extract(url, save_path, extract_path):
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

    print(f"Extracting {save_path}...")
    try:
        extract_kwargs = {'path': extract_path}
        if sys.version_info >= (3, 12):
            # Safe extraction filters were added in Python 3.12
            extract_kwargs['filter'] = 'data'
        with tarfile.open(save_path) as tar:
            tar.extractall(**extract_kwargs)
    except Exception as e:
        print(f"Failed to extract: {e}")
        return False

    os.remove(save_path)
    print("Download and extraction complete.")
    return True


def download_from_hf(category, data_dir):
    """Fallback: fetch <category>/** from the Hugging Face dataset mirror."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed (pip install huggingface_hub).")
        return False

    print(f"Downloading '{category}' from Hugging Face mirror ({HF_REPO})...")
    try:
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=[f"{category}/**"],
            local_dir=data_dir,
        )
    except Exception as e:
        print(f"Hugging Face download failed: {e}")
        return False

    if not os.path.isdir(os.path.join(data_dir, category)):
        print(f"Category '{category}' not found in the HF mirror.")
        return False
    print("Download complete.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MVTec AD subset")
    parser.add_argument('--category', type=str, default='bottle', help='Category to download')
    parser.add_argument('--data_dir', type=str, default='data/mvtec', help='Directory to save data')

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    target_dir = os.path.join(args.data_dir, args.category)
    if os.path.exists(target_dir):
        print(f"Category '{args.category}' already exists at {target_dir}")
        sys.exit(0)

    ok = False
    if args.category in MIRRORS:
        url = MIRRORS[args.category]
        save_path = os.path.join(args.data_dir, f"{args.category}.tar.xz")
        ok = download_and_extract(url, save_path, args.data_dir)

    if not ok:
        ok = download_from_hf(args.category, args.data_dir)

    if not ok:
        print(f"Automatic download failed. Please download manually from {MANUAL_URL}")
        print(f"And extract it into {target_dir}")
        sys.exit(1)
