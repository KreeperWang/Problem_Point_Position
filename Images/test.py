#!/usr/bin/env python3
"""
download_blip2.py

A script to download specified files of Salesforce/blip2-opt-2.7b
from Tsinghua University Mirror of Hugging Face Hub for offline use.

It downloads the following files:
  - .gitattributes
  - README.md
  - added_tokens.json
  - config.json
  - generation_config.json
  - merges.txt
  - model-00001-of-00002.safetensors
  - model-00002-of-00002.safetensors
  - model.safetensors.index.json
  - preprocessor_config.json
  - processor_config.json
  - pytorch_model-00001-of-00002.bin
  - pytorch_model-00002-of-00002.bin
  - pytorch_model.bin.index.json
  - special_tokens_map.json
  - tokenizer.json
  - tokenizer_config.json
  - vocab.json

Usage:
    python download_blip2.py [--output_dir /path/to/blip2-opt-2.7b]
"""
import argparse
import sys
import os
# 在训练脚本开头添加
import numpy as np


def download_file(url: str, dest_path: str):
    """
    Download a file from given URL to dest_path with streaming.
    """
    try:
        import requests
    except ImportError:
        print("requests not installed. Installing...", file=sys.stderr)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    print(f"Downloading {url} to {dest_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        done = int(50 * downloaded / total)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}MB/{total/1024/1024:.2f}MB")
                        sys.stdout.flush()
        if total:
            print()


def main():
    print(f"Numpy version: {np.__version__}")  # 验证 numpy 可用
    parser = argparse.ArgumentParser(
        description="Download specified files via Tsinghua HF mirror"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"F:\\bnu_task\\summer_camp_mllm_task_train\\Problem_Point_Position_train\\model",
        help="Directory to store the downloaded files"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    repo_id = "Salesforce/blip2-opt-2.7b"
    base_url = "https://huggingface.co"
    filenames = [
        ".gitattributes",
        "README.md",
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "processor_config.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json"
    ]

    for fname in filenames:
        file_url = f"{base_url}/{repo_id}/resolve/main/{fname}"
        dest_file = os.path.join(args.output_dir, fname)
        try:
            download_file(file_url, dest_file)
            print(f"✔ Successfully downloaded {fname}")
        except Exception as e:
            print(f"❌ Error downloading {fname}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"All files downloaded to {args.output_dir}")

if __name__ == '__main__':
    main()
