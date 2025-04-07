#!/usr/bin/env python3
import os
import argparse
import torch

from safetensors import safe_open

def print_tensor_tsv(model_dir, depth):
    '''Print tensor info in .safetensors into tsv format'''
    TENSOR_CLASS = {
        'weight_scale_inv': 'scale',
        'weight_scale': 'scale'
    }
    print('SafetensorsFile\tTensorKey\tTensorParams\tTensorType\tTensorShape')
    safetensor_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
    summary = {}
    for filename in safetensor_files:
        file_path = os.path.join(model_dir, filename)
        with safe_open(file_path, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                print(f'{filename}\t{key}\t{tensor.numel()}\t{tensor.dtype}\t{tensor.shape}')
                lst = key.split('.')
                # Get suffix: .weight or .weight_scale_inv
                tclass = TENSOR_CLASS[lst[-1]] if lst[-1] in TENSOR_CLASS else 'weight'
                # Limit prefix to dep
                dep = min(len(lst), depth+1) if depth > 0 else len(lst)
                # Get summary of prefixes
                for prefix in ['.'.join(lst[:i]) for i in range(0, dep)]:
                    summary[f'{tclass}[{prefix}]'] = summary.get(f'{tclass}[{prefix}]', 0) + tensor.numel()
    for key in sorted(summary):
        print(f'Summary\t{key}\t{summary[key]}\t\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print tensor shape and dtype of .safetensors file')
    parser.add_argument('model_dir', nargs='?', default='.', help='Model directory (default: $PWD)')
    parser.add_argument('--summary_depth', '-d', type=int, default=3, help='Summary depth of weights')
    args = parser.parse_args()
    print_tensor_tsv(args.model_dir, args.summary_depth)
