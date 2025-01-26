#!/usr/bin/env python3
import torch
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_adapters(base_model_path, adapter_path, output_model_path):
    start = time.time()
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=True  # Set to True if the model uses custom code
    )

    # Load the LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map={"": "cpu"},
    )

    # Merge the LoRA adapters into the base model
    print("Merging LoRA adapters into the base model...")
    with torch.no_grad():                # Ensure no gradients are tracked
        model = model.merge_and_unload()

    # Save the merged model
    # Save the merged model
    print(f"Saving the merged model to '{output_model_path}'...")
    model.save_pretrained(output_model_path)
    base_model.save_pretrained(output_model_path)  # Ensure all necessary files are saved

    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_model_path)

    print(f"Merged model saved to {output_model_path}")
    
    hours, rem = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == "__main__":
    base_model_path = "DeepSeek-R1-Distill-Llama-8B"
    adapter_path = "fine_tuned_llama8b"
    output_model_path = "merged_llama8b"

    merge_lora_adapters(base_model_path, adapter_path, output_model_path)

