"""
Automated downloader and extractor for KuaiRec and KuaiRand datasets from Zenodo.

Downloads:
  - KuaiRec: https://zenodo.org/records/18164998/files/KuaiRec.zip
  - KuaiRand (Pure variant): https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz

Usage:
    python download_kuai.py [--dataset {all,kuairec,kuairand}]
"""

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

URLS = {
    "kuairec": {
        "url": "https://zenodo.org/records/18164998/files/KuaiRec.zip",
        "archive_name": "KuaiRec.zip",
        "target_dir": ROOT / "data" / "kuairec",
        "format": "zip",
    },
    "kuairand": {
        "url": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
        "archive_name": "KuaiRand-Pure.tar.gz",
        "target_dir": ROOT / "data" / "kuairand",
        "format": "tar.gz",
    },
}


def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {percent:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)")
    else:
        sys.stdout.write(f"\r  Downloading: {downloaded / (1024 * 1024):.1f} MB")
    sys.stdout.flush()


def download_and_extract(dataset_key: str):
    info = URLS[dataset_key]
    target_dir = info["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / info["archive_name"]

    print(f"\n[{dataset_key.upper()}] Starting download...")
    print(f"  URL: {info['url']}")
    print(f"  Destination: {archive_path}")

    # Check if already extracted
    data_subdir = target_dir / "data"
    if data_subdir.exists() and any(data_subdir.iterdir()):
        print(f"  ✓ {dataset_key} data already appears to be extracted at {data_subdir}. Skipping.")
        return

    # Download if archive doesn't exist
    if not archive_path.exists():
        try:
            # Provide headers to avoid Zenodo blocking automated agents
            req = urllib.request.Request(
                info["url"],
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', -1))
                block_size = 1024 * 1024  # 1MB blocks
                downloaded = 0
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = min(100.0, downloaded * 100.0 / total_size)
                        sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)")
                    else:
                        sys.stdout.write(f"\r  Progress: {downloaded/(1024*1024):.1f} MB")
                    sys.stdout.flush()
            print("\n  ✓ Download complete.")
        except Exception as e:
            print(f"\n  ❌ Failed to download {dataset_key}: {e}")
            print(f"  Please download manually from: {info['url']}")
            print(f"  and place it in: {archive_path}")
            return
    else:
        print(f"  ✓ Archive {archive_path.name} already exists.")

    # Extract
    print(f"  Extracting {archive_path.name} to {target_dir} ...")
    try:
        if info["format"] == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif info["format"] == "tar.gz":
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(target_dir)
        print(f"  ✓ Extraction complete!")
    except Exception as e:
        print(f"  ❌ Error during extraction: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract KuaiRec and KuaiRand datasets.")
    parser.add_argument("--dataset", choices=["all", "kuairec", "kuairand"], default="all",
                        help="Which dataset to download.")
    args = parser.parse_args()

    targets = ["kuairec", "kuairand"] if args.dataset == "all" else [args.dataset]
    for key in targets:
        download_and_extract(key)

    print("\nNext step: Run preprocessing:")
    print("  python preprocess_kuairec.py --n_clusters 50")
    print("  python preprocess_kuairand.py --n_clusters 50 --variant pure")


if __name__ == "__main__":
    main()
