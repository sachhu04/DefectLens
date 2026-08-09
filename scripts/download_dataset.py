import os
import sys
import tarfile
import urllib.request
import argparse

# Some community mirrors for MVTec AD categories (these can sometimes expire)
MIRRORS = {
    'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz'
}

def download_and_extract(url, save_path, extract_path):
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Please download manually from https://www.mvtec.com/company/research/datasets/mvtec-ad")
        return False
        
    print(f"Extracting {save_path}...")
    try:
        with tarfile.open(save_path) as tar:
            tar.extractall(path=extract_path)
    except Exception as e:
        print(f"Failed to extract: {e}")
        return False
        
    os.remove(save_path)
    print("Download and extraction complete.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MVTec AD subset")
    parser.add_argument('--category', type=str, default='bottle', help='Category to download')
    parser.add_argument('--data_dir', type=str, default='data/mvtec', help='Directory to save data')
    
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.category in MIRRORS:
        url = MIRRORS[args.category]
        save_path = os.path.join(args.data_dir, f"{args.category}.tar.xz")
        
        target_dir = os.path.join(args.data_dir, args.category)
        if os.path.exists(target_dir):
            print(f"Category '{args.category}' already exists at {target_dir}")
        else:
            download_and_extract(url, save_path, args.data_dir)
    else:
        print(f"No direct mirror configured for '{args.category}'.")
        print("Please download manually from https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print(f"And extract it into {args.data_dir}/{args.category}")
