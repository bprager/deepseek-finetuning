import os
import json
import time
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

def main():
    # ------------------------------
    # 1. Configuration Parameters
    # ------------------------------
    MODEL_NAME_OR_PATH = "DeepSeek-R1-Distill-Llama-8B"  # Updated to 8B model
    TRAIN_DATA_FILE = "political_bias_fine_tuning_data.json"
    OUTPUT_DIR = "./fine_tuned_llama8b"
    LOGGING_DIR = "./logs"  # Directory for TensorBoard logs
    SEED = 42
    NUM_EPOCHS = 3
    BATCH_SIZE = 1  # Reduced from 2 to save memory
    LEARNING_RATE = 5e-5
    LOGGING_STEPS = 50  # Increased frequency for better monitoring
    SAVE_STEPS = 500
    MAX_SEQ_LENGTH = 64  # Reduced from 128 to save memory
    WARMUP_STEPS = 0  # Set to desired value if using warmup

    # ------------------------------
    # 2. Start Timer
    # ------------------------------
    start_time = time.time()

    # ------------------------------
    # 3. Clean Up GPU Memory
    # ------------------------------
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------
    # 4. Set Seed for Reproducibility
    # ------------------------------
    set_seed(SEED)

    # ------------------------------
    # 5. Load the Tokenizer
    # ------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        use_fast=True
    )

    # ------------------------------
    # 6. Load the Model with 8-bit Quantization
    # ------------------------------
    print("Loading model with 8-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True  # Set to True if the model uses custom code
    )

    # ------------------------------
    # 7. Prepare the Dataset
    # ------------------------------
    print("Loading and processing dataset...")
    # Load the JSON data
    with open(TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON data should be a list of dictionaries.")

    if len(data) == 0:
        raise ValueError("JSON data is empty.")

    # Print the keys of the first entry for verification
    first_entry = data[0]
    print("Keys in the first data entry:", first_entry.keys())

    # Define the input and target keys based on your data
    INPUT_KEY = 'instruction'
    TARGET_KEY = 'output'

    # Verify that the required keys exist in all entries
    for idx, entry in enumerate(data):
        if INPUT_KEY not in entry:
            raise KeyError(f"Entry {idx} is missing the '{INPUT_KEY}' key.")
        if TARGET_KEY not in entry:
            raise KeyError(f"Entry {idx} is missing the '{TARGET_KEY}' key.")

    print(f"Using '{INPUT_KEY}' as input and '{TARGET_KEY}' as target.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(data)

    # Split the dataset into training and validation sets (90% train, 10% eval)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    # Tokenization Function
    def tokenize_function(examples):
        """
        Tokenizes the input and target texts.
        If the 'input' field is non-empty, it is included between 'instruction' and 'output'.
        """
        instructions = examples[INPUT_KEY]
        outputs = examples[TARGET_KEY]
        inputs = examples.get('input', [''] * len(instructions))  # Default to empty strings if 'input' key is missing

        # Construct the prompt. If 'input' is non-empty, include it.
        prompts = []
        for instr, inp, out in zip(instructions, inputs, outputs):
            if inp.strip():
                prompt = f"{instr}\nInput: {inp}\nOutput: {out}"
            else:
                prompt = f"{instr}\nOutput: {out}"
            prompts.append(prompt)

        return tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

    # Apply tokenization to the training and evaluation datasets
    print("Tokenizing training dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    print("Tokenizing evaluation dataset...")
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # ------------------------------
    # 8. Data Collator
    # ------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )

    # ------------------------------
    # 9. Parameter-Efficient Fine-Tuning with LoRA
    # ------------------------------
    print("Configuring LoRA for parameter-efficient fine-tuning...")
    lora_config = LoraConfig(
        r=1,  # Reduced from 2 to save memory
        lora_alpha=4,  # Reduced from 8
        lora_dropout=0.0125,  # Reduced from 0.025
        target_modules=["q_proj", "v_proj"],  # Ensure these are correct for LLaMA-8B
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()  # Ensure model is in training mode

    # Diagnostic: Verify Trainable Parameters
    print("----- Trainable Parameters -----")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    print("----- End of Trainable Parameters -----")

    # ------------------------------
    # 10. Define Training Arguments Without DeepSpeed
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=32,  # Adjusted based on reduced batch size
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=LOGGING_STEPS,
        log_level="debug",
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision
        optim="adamw_torch",
        push_to_hub=False,
        gradient_checkpointing=False,  # Disabled to prevent conflicts
        disable_tqdm=False,
        report_to=["tensorboard"],  # Enable TensorBoard
        # deepspeed="deepspeed_config.json"  # Removed DeepSpeed
        # warmup_steps=1000,  # Uncomment if using warmup
    )

    # ------------------------------
    # 11. Initialize Trainer
    # ------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator
        # Removed 'callbacks' parameter to avoid duplication
    )

    # ------------------------------
    # 12. Start Fine-Tuning
    # ------------------------------
    print("Starting fine-tuning...")
    trainer.train()

    # ------------------------------
    # 13. Save the Fine-Tuned Model
    # ------------------------------
    print("Saving the fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning complete.")

    # ------------------------------
    # 14. Calculate and Print Time Taken
    # ------------------------------
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken: %d:%02d:%02d" % (int(hours), int(minutes), int(seconds)))

if __name__ == "__main__":
    main()
