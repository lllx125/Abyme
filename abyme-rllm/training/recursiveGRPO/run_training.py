import os
import gc
import torch
import dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
from abyme.magic import abyme_system_prompt

dotenv.load_dotenv()

# ==========================================
# HYPERPARAMETERS
# ==========================================
LORA_RANK = 64
LORA_ALPHA = 128
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 32  # 2 * 32 = 64 effective batch size
LEARNING_RATE = 2e-6              # RL requires lower learning rates than SFT
NUM_EPOCHS = 1
BETA = 0.04                       # KL Divergence Penalty coefficient
MAX_SEQ_LENGTH = 4096 


# ==========================================
# CUSTOM OFFLINE GRPO TRAINER
# ==========================================
class OfflineGRPOTrainer(Trainer):
    def __init__(self, beta=0.04, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom Policy Gradient loss using pre-computed advantages and 
        on-the-fly KL penalty via adapter toggling.
        """
        # 1. Extract Advantages (Convert to tensor if not already)
        advantages = inputs.pop("advantages")
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = advantages.to(model.device)

        labels = inputs["labels"]
        
        # 2. Forward pass for Policy Model (pi_theta)
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 3. Forward pass for Reference Model (pi_ref) - VRAM SAVER TRICK
        with model.disable_adapter():
            with torch.no_grad():
                ref_outputs = model(**inputs)
                ref_logits = ref_outputs.logits

        # 4. Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 5. Calculate per-token Log Probabilities
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_nll = token_nll.view(shift_labels.size())
        
        ref_token_nll = loss_fct(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_labels.view(-1))
        ref_token_nll = ref_token_nll.view(shift_labels.size())

        # Mask out padding (-100)
        mask = (shift_labels != -100)
        
        # Sum over sequence length (dim=1) to get full sequence log probability
        log_pi_theta = -(token_nll * mask).sum(dim=1)
        log_pi_ref = -(ref_token_nll * mask).sum(dim=1)

        # 6. Calculate Clipped GRPO/PPO Objective
        ratio = torch.exp(log_pi_theta - log_pi_ref)
        clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        # 7. Calculate Exact KL Divergence (Approximate PPO style)
        kl = torch.exp(log_pi_ref - log_pi_theta) - (log_pi_ref - log_pi_theta) - 1.0

        # 8. Total Loss
        loss = (policy_loss + self.beta * kl).mean()

        return (loss, outputs) if return_outputs else loss


# ==========================================
# TRAINING EXECUTION
# ==========================================
def run_training(base_model_path: str, local_jsonl_path: str, output_adapter_dir: str, resume: bool = False):
    """
    Trains the LoRA adapters on the offline GRPO data and saves them locally.
    Does NOT push to hub.
    """
    print(f"Loading base model: {base_model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          
        load_in_4bit=False,   
    )

    # Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
    )
    
    if resume and os.path.exists(output_adapter_dir):
        print(f"Resuming training from existing adapter: {output_adapter_dir}")
        model.load_adapter(output_adapter_dir)
        # Ensure it remains trainable
        for param in model.parameters():
            if param.requires_grad is False and "lora" in param.name: # Only set LoRA layers
                param.requires_grad = True

    print(f"Loading local dataset: {local_jsonl_path}")
    dataset = load_dataset("json", data_files=local_jsonl_path, split="train").shuffle(seed=42)

    # ----------------------------------------------------
    # Data Formatting: Prompt masking + Advantage injection
    # ----------------------------------------------------
    def format_grpo_dataset(example):
        prompt_msg = [
            {"role": "system", "content": abyme_system_prompt},
            {"role": "user", "content": example["input"]}
        ]
        completion_msg = [{"role": "assistant", "content": example["output"]}]
        
        prompt_str = tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)
        completion_str = tokenizer.apply_chat_template(completion_msg, tokenize=False, add_generation_prompt=False)
        
        if tokenizer.bos_token:
            completion_str = completion_str.replace(tokenizer.bos_token, "")
            
        full_text = prompt_str + completion_str
        
        # Tokenize full text
        encoded = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
        # Tokenize prompt to find the masking boundary
        prompt_encoded = tokenizer(prompt_str, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
        
        labels = encoded["input_ids"].copy()
        prompt_len = len(prompt_encoded["input_ids"])
        
        # Mask the prompt so we only compute loss on the generation
        labels[:prompt_len] = [-100] * prompt_len
        
        encoded["labels"] = labels
        encoded["advantages"] = example["advantage"]
        return encoded

    # Map dataset and keep ONLY the columns the model needs
    dataset = dataset.map(format_grpo_dataset, remove_columns=dataset.column_names)

    # ----------------------------------------------------
    # Trainer Setup
    # ----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_adapter_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=10,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        remove_unused_columns=False, # CRITICAL: Prevents HF from discarding our "advantages" column
        report_to="none",
    )

    trainer = OfflineGRPOTrainer(
        beta=BETA,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    # ----------------------------------------------------
    # Train & Save Locally
    # ----------------------------------------------------
    print("Starting Offline GRPO Training...")
    trainer.train()

    print(f"Training complete! Saving LoRA adapters to {output_adapter_dir}")
    model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)

    # Free memory
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ==========================================
# CURRICULUM UPLOAD UTILITY
# ==========================================
def merge_and_upload(base_model_path: str, lora_adapter_dir: str, hub_repo_id: str):
    """
    Loads the base model and the trained LoRA adapters, merges them into 16-bit,
    and pushes the final model to the Hugging Face Hub.
    """
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is missing. Cannot upload to Hub.")

    print(f"Merging LoRA adapters from {lora_adapter_dir} into base model {base_model_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )

    # Load the locally saved adapters onto the base model
    model.load_adapter(lora_adapter_dir)

    print(f"Pushing merged model to Hugging Face Hub: {hub_repo_id}")
    model.push_to_hub_merged(
        hub_repo_id,
        tokenizer=tokenizer,
        save_method="merged_16bit",
        token=HF_TOKEN,
    )

    # Push a clean tokenizer to avoid unsloth backend warnings
    clean_tokenizer = AutoTokenizer.from_pretrained(hub_repo_id, token=HF_TOKEN)
    clean_tokenizer.push_to_hub(hub_repo_id, token=HF_TOKEN)

    print(f"Success! Model uploaded to: https://huggingface.co/{hub_repo_id}")

    del model, tokenizer, clean_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    BASE_MODEL = "Lixing-Li/Abyme-Qwen3.5-9B-SFT"
    LOCAL_DATA = "results/grpo_0_train.jsonl"
    ADAPTER_OUTPUT_DIR = "checkpoints/abyme-grpo-iteration-0"
    HUB_REPO = "Lixing-Li/Abyme-Qwen3.5-9B-GRPO-Live"

    # Step 1: Train and save locally (Call this in a loop for your curriculum)
    run_training(
        base_model_path=BASE_MODEL,
        local_jsonl_path=LOCAL_DATA,
        output_adapter_dir=ADAPTER_OUTPUT_DIR
    )

    # Step 2: Merge and upload (Call this every N iterations)
    # merge_and_upload(
    #     base_model_path=BASE_MODEL,
    #     lora_adapter_dir=ADAPTER_OUTPUT_DIR,
    #     hub_repo_id=HUB_REPO
    # )