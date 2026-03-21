import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from abyme.magic import abyme_system_prompt

# ==========================================
# 1. CONFIGURATION & LOGIN
# ==========================================
# Ensure you export your HF_TOKEN in your terminal before running:
# export HF_TOKEN="your_token_here"
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable.")

login(token=HF_TOKEN, add_to_git_credential=True)

HF_ORG = "Lixing-Li"
DATASET_HUB_NAME = f"{HF_ORG}/Abyme-finetune-dataset-formatted"
MODEL_HUB_NAME = f"{HF_ORG}/Abyme-Llama-3.1-8B-SFT"

max_seq_length = 4096

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Apply the Llama 3.1 chat template natively
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# ==========================================
# 3. LoRA ADAPTERS (Your Hyperparameters)
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

def formatting_func(examples):
    texts = []
    for i in range(len(examples["input"])):
        messages = [
            {"role": "system", "content": abyme_system_prompt.strip()},
            {"role": "user", "content": examples["input"][i]},
            {"role": "assistant", "content": examples["output"][i]},
        ]
        
        # apply_chat_template handles adding the proper Llama-3 tokens
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

dataset = load_dataset(DATASET_HUB_NAME, split="train")
dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

# ==========================================
# 5. TRAINER (Your Hyperparameters)
# ==========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,  # Uncontaminated packing enabled!
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=3e-5,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# Apply response masking over the packed dataset
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# ==========================================
# 6. TRAIN & SAVE
# ==========================================
trainer.train()

model.push_to_hub_merged(
    MODEL_HUB_NAME,
    tokenizer,
    save_method="merged_16bit",
    token=HF_TOKEN
)