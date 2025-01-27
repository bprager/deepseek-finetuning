#!/usr/bin/env python3
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import torch
import time

def convert_to_safetensors(model_path, output_path):
    start = time.time()
    # Load the merged model onto the CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # Changed from "auto" to "cpu"
        trust_remote_code=True
    )

    # Get the state dictionary
    state_dict = model.state_dict()

    # Save using safetensors
    save_file(state_dict, output_path)
    print(f"Model converted to safetensors and saved at {output_path}")
    
    hours, rem = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == "__main__":
    model_path = "merged_llama8b"
    output_path = "merged_llama8b/safetensors_model.safetensors"

    convert_to_safetensors(model_path, output_path)
