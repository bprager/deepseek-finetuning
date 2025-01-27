# deepseek-finetuning

Fine-tuning of the Open Source deepseek model with local resources

## Environment

The fine-tuning is done with a local Ollama environment
running on a AMD Ryzen 9 7950X 16-Core processor, 61GB Ram,
GeForce RTX 2060 (16GB VRAM)

## Goal

This is an exercise to fine-tune th eopen source deepseek model.

One purpose is to overcome bias introduced by being a Chinese model
e.g. political influence of the CCP on topics like Taiwan, Tiananmen
Square protest, etc.

## Steps

- Generate a 3,000 question-/answer- dataset via a Microsoft psi4 model.
- Identify a suitable model from Huggingface
  (e.g. https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- Fine-tunes the "deepseek" model using a LoRA-based parameter-efficient method
  to produce a smaller adapter while preserving overall performance.
 - Merge the adapter with the base model
 - Convert to `safetensors` for use with Ollama
- Create the new Ollama model `Collama create fine_tuned_deepseek`
- Quantize the new model `ollama create --quantize q4_K_M fine_tuned_deepseek`
