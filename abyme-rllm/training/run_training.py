import torch
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
import os
import dotenv
from training.notifier import DiscordNotifier

dotenv.load_dotenv()


# 1. Patch Unsloth for Preference Trainers (Must be called before initializing Trainer)
PatchDPOTrainer()

# ==========================================
# CONFIGURATION
# ==========================================

LORA_RANK = 64
LORA_ALPHA = 128
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1

def run_training(model_name, dataset_id, hub_repo_id):
    HF_TOKEN = os.getenv("HF_TOKEN", "")                      # HuggingFace Write Token

    MAX_SEQ_LENGTH = 2048 

    # ==========================================
    # LOAD MODEL & TOKENIZER
    # ==========================================
    # We use load_in_4bit=True by default to save VRAM. 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # Auto-detects bfloat16 for A100
        load_in_4bit=False,   
        token=HF_TOKEN,
    )

    # Apply LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", # Crucial for long-context memory saving
        random_state=3407,
    )

    # ==========================================
    # PREPARE DATASET
    # ==========================================
    dataset = load_dataset(dataset_id, split="train")

    # TRL's KTOTrainer expects 'prompt', 'completion', and a boolean 'label'
    def format_kto_dataset(example):
        # Format input as a user prompt
        prompt_msg = [{"role": "user", "content": example["input"]}]
        # Format output as the assistant completion
        completion_msg = [{"role": "assistant", "content": example["output"]}]
        
        return {
            "prompt": tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True),
            # Remove the BOS token from completion if the tokenizer adds it, to prevent double BOS
            "completion": tokenizer.apply_chat_template(completion_msg, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, ""),
            "label": bool(example["label"]) # Ensure label is strictly True/False
        }

    # Map and remove old columns to avoid Trainer confusion
    dataset = dataset.map(format_kto_dataset, remove_columns=dataset.column_names)

    # ==========================================
    # KTO TRAINER SETUP
    # ==========================================
    kto_args = KTOConfig(
        per_device_train_batch_size=BATCH_SIZE,   # A100 can handle larger batches; bump to 16 if VRAM allows
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=0.1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,              # Preference tuning uses lower LRs than standard SFT
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),    # Will automatically trigger True on your A100
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="kto_outputs",
        beta=0.1,                        # KTO implicit reward margin hyperparameter
    )

    trainer = KTOTrainer(
        model=model,
        ref_model=None, # Unsloth + PEFT implicitly handles the reference model natively
        args=kto_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # ==========================================
    # TRAIN & PUSH MERGED MODEL
    # ==========================================
    mailman.send("Starting KTO Fine-tuning...")
    trainer.train()

    print("Training complete! Merging LoRA adapters into 16-bit base model and pushing to Hugging Face...")
    # save_method="merged_16bit" fuses the adapter weights directly into the base model's weights.
    model.push_to_hub_merged(
        hub_repo_id=hub_repo_id,
        tokenizer=tokenizer,
        save_method="merged_16bit",
        token=HF_TOKEN,
    )
    print(f"Success! Model uploaded to: https://huggingface.co/{hub_repo_id}")
    mailman.send(f"Success! Model uploaded to: https://huggingface.co/{hub_repo_id}")