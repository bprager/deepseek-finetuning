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

- Generate a 3,000 question-/answer- dataset via a MS psi4 model.
- Define a fine-tuning aproach
