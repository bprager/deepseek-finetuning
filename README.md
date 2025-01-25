# deepseek-finetuning

Fine-tuning of the Open Source deepseek model

## Environment

The fine-tuning is done with a local Ollama environment
running on a AMD Ryzen 9 16-Core processor and a GeForce RTX 2060.

## Goal

This is an exercise to fine-tune th eopen source deepseek model.

One purpose is to overcome bias introduced by being a Chinese model
e.g. political influence of the CCP on topics like Taiwan, Tiananmen
Square protest, etc.

## Steps

- Generate a 3,000 question-/answer- dataset via a Microsoft psi4 model.
- Identify a suitable model from Huggingface
  (e.g. https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B)
- Fine-tunes the "deepseek" model using a LoRA-based parameter-efficient method
  to produce a smaller adapter while preserving overall performance.
