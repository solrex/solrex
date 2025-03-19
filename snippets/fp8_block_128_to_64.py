#!/bin/env python3
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import json

import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

def main(b128_path, b64_path, model_name="deepseek-ai/DeepSeek-R1"):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(b64_path, exist_ok=True)
    model_index_file = os.path.join(b64_path, "model.safetensors.index.json")
    config_file = os.path.join(b64_path, "config.json")
     
    if not os.path.exists(model_index_file) or not os.path.exists(config_file):
        snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.safetensors"],
            local_dir=b64_path,
            local_dir_use_symlinks=False
        )
        print(f"model index file and config file downloaded to {b64_path}")

        # modify config.json and save it
        config = json.load(open(config_file))
        # modify block size from 128x128 to 64x64
        quant_config = config["quantization_config"]
        quant_config["weight_block_size"] = [
            64,
            64
        ]
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"config.json modified and saved to {config_file}")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    safetensor_files = list(glob(os.path.join(b128_path, "*.safetensors")))
    safetensor_files.sort()
    safetensor_files = safetensor_files[130:]
    quant_count = 0
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cuda")
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            scale_inv_name = f"{weight_name}_scale_inv"
            if weight_name.endswith('_scale_inv'):
                print(f'Expanding: {weight_name=}')
                assert weight.element_size() == 4
                assert weight.dim() == 2
                quant_count += 1
                expand_weight = torch.repeat_interleave(weight, repeats=2, dim=0)
                expand_weight = torch.repeat_interleave(expand_weight, repeats=2, dim=1)
                new_state_dict[weight_name] = expand_weight
            else:
                new_state_dict[weight_name] = weight
        new_safetensor_file = os.path.join(b64_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
    print(f"{quant_count} weights are expanded to block 64x64.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-b128-hf-path", '-i', type=str, required=True)
    parser.add_argument("--output-b64-hf-path", '-o', type=str, required=True)
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1")
    args = parser.parse_args()
    main(args.input_b128_hf_path, args.output_b64_hf_path, args.model_name)
    print("done")
