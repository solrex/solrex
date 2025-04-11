#!/bin/env python3
import sys
import argparse
import torch

from transformers import AutoTokenizer, TextStreamer
# Change the model if needed
from transformers import AutoModelForCausalLM as TheModel
#from transformers import Qwen2_5_VLForConditionalGeneration as TheModel

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def apply_fp8_quant_to_llm(model_dir, output_dir, device_map, max_sample_token):
    if output_dir == None:
        output_dir = model_dir.rstrip("/") + "-FP8-Dynamic"

    # Load model.
    model = TheModel.from_pretrained(
        model_dir, device_map=device_map, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Confirm generations of the quantized model look sane.
    prompt = "请输出《滕王阁序》全文"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("========== SAMPLE BEFORE QUANT ==============")
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        streamer=streamer,
        max_new_tokens=max_sample_token)
    print("==========================================")
    
    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp8 with per channel via ptq
    #   * quantize the activations to fp8 with dynamic per token
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )
    
    # Apply quantization and save to disk in compressed-tensors format.
    oneshot(model=model, recipe=recipe, tokenizer=tokenizer, output_dir=output_dir)
    tokenizer.save_pretrained(output_dir)

    # Load output model
    del model
    model = TheModel.from_pretrained(
        output_dir, device_map=device_map, torch_dtype=torch.bfloat16
    )
    del tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE AFTER QUANT ==============")
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        streamer=streamer,
        max_new_tokens=max_sample_token)
    print("==========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do FP8-Dynamic quant for model')
    parser.add_argument('model_dir', nargs='?', default='.', help='Model directory (default: $PWD)')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='Output directory (default: $model_dir-FP8-Dynamic')
    parser.add_argument('--device_map', '-d', type=str, default='auto', help='device_map, can be: auto, cuda:X, balanced, sequential, default: auto')
    parser.add_argument('--max_sample_token', '-m', type=int, default=400, help='max output token in sampling before and after quant, default: 400')
    args = parser.parse_args()
    apply_fp8_quant_to_llm(args.model_dir, args.output_dir, args.device_map, args.max_sample_token)
