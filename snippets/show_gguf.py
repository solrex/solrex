#!/usr/bin/env python3
import os
import argparse
import torch

from gguf import GGUFReader, GGUFValueType, ReaderTensor

def print_tensor_tsv(model_dir, depth):
    '''Print tensor info in .safetensors into tsv format'''
    TENSOR_CLASS = {
        'weight': 'weight',
        'bias': 'weight',
        'weight_scale_inv': 'scale'
    }
    print('SafetensorsFile\tTensorKey\tTensorParams\tTensorType\tTensorShape')
    safetensor_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.gguf')])
    summary = {}
    for filename in safetensor_files:
        file_path = os.path.join(model_dir, filename)
        reader = GGUFReader(file_path, 'r')
        for n, tensor in enumerate(reader.tensors, 1):
            print(f'{filename}\t{tensor.name}\t{tensor.n_elements}\t{tensor.tensor_type.name}\t{tensor.shape}')
            lst = tensor.name.split('.')
            # Get suffix: .weight or .weight_scale_inv
            tclass = TENSOR_CLASS[lst[-1]] if lst[-1] in TENSOR_CLASS else lst[-1]
            # Limit prefix to dep
            dep = min(len(lst), depth+1) if depth > 0 else len(lst)
            # Get summary of prefixes
            for prefix in ['.'.join(lst[:i]) for i in range(0, dep)]:
                summary[f'{tclass}[{prefix}]'] = summary.get(f'{tclass}[{prefix}]', 0) + tensor.n_elements
    for key in sorted(summary):
        print(f'Summary\t{key}\t{summary[key]}\t\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print tensor shape and dtype of .gguf file')
    parser.add_argument('model_dir', nargs='?', default='.', help='Model directory (default: $PWD)')
    parser.add_argument('--summary_depth', '-d', type=int, default=3, help='Summary depth of weights')
    args = parser.parse_args()
    print_tensor_tsv(args.model_dir, args.summary_depth)
