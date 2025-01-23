#!/usr/bin/env python3
import os
import argparse
import torch

from safetensors import safe_open

def print_tensor_tsv(model_dir):
    """Print tensor info in .safetensors into tsv format"""
    print("SafetensorsFile\tTensorKey\tTensorShape\tTensorType")
    safetensor_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
    for filename in safetensor_files:
        file_path = os.path.join(model_dir, filename)
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                print(f"{filename}\t{key}\t{tensor.shape}\t{tensor.dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print tensor shape and dtype of .safetensors file")
    parser.add_argument("model_dir", nargs='?', default='.', help="Model directory (default: $PWD)")
    args = parser.parse_args()
    print_tensor_tsv(args.model_dir)
