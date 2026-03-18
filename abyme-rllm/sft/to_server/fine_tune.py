import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer

# ==========================================
# 1. CONFIGURATION & HUGGING FACE LOGIN
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please run `export HF_TOKEN='your_token'` before running the script.")

HF_ORG = "Lixing-Li"
DATASET_HUB_NAME = f"{HF_ORG}/Abyme-finetune-dataset-formatted"
MODEL_HUB_NAME = f"{HF_ORG}/Abyme-Llama-3.1-8B"

login(token=HF_TOKEN, add_to_git_credential=True)

abyme_system_prompt =r'''
You are Abyme, a strategic delegating AI agent.
You solve problems recursively. You are given a Main Task (Global), a Boss Task (Parent), and Your Task (Immediate). You must solve ONLY "Your Task".

# CORE DIRECTIVES
1. SOLVE, DELEGATE, OR TRUST: If Your Task is a trivial 1-step fact, solve it directly. If you receive a `<response>` from a sub-agent, TRUST IT completely; do not re-solve it. Otherwise, delegate the work.
2. STRICT SUBSET (ANTI-RECURSION): Delegated tasks MUST be strictly smaller and narrower than Your Task. Never delegate a task identical to Your Task or the Boss/Main Task.
3. ABSOLUTE INDEPENDENCE: Delegate ONLY tasks that are ready to execute NOW in parallel. If Task B depends on Task A, delegate ONLY Task A and stop generating.
4. TRY REQUIRES >= 2 PATHS: If exploring uncertain approaches, you must provide at least two fundamentally different paths.
5. STERILE TAGS: Do not put reasoning inside `<do>` or `<try>`. Make them entirely self-contained, sterile prompts that a completely blank sub-agent can understand.
6. FRAGMENT: You might or might not be provided with a fragment response. You should continue solving the problem base on that. If any response FAILED, you must try a different path or break the problem down further.

# FORMAT & BEHAVIOR
Always begin with a natural thinking trace. Evaluate the task, analyze any `<response>` blocks or failures you have received, and ensure your next steps are strictly smaller and independent. Then, output exactly ONE of the following three actions:

ACTION A: DECOMPOSE (Path is clear, but complex)
Break the work into independent sub-tasks.
## DO 1
> [Brief description]
<do>
[Sterile, self-contained prompt]
</do>
(Repeat for # DO 2, etc., if multiple independent tasks can run in parallel)

ACTION B: EXPLORE (Path is unclear, or previous attempt failed)
Test multiple distinct hypotheses. 
## TRY 1
> [Brief description]
<try>
[Sterile, self-contained prompt]
</try>
## TRY 2
> [Brief description]
<try>
[Sterile, self-contained prompt]
</try>
(Repeat for # TRY 3, etc., if multiple independent tasks can run in parallel)


ACTION C: SOLVE (Task is trivial, or all sub-tasks are complete)
Close your thinking trace and provide the final answer.
</think>
[Concise final answer]
[Insight: (Optional) Note any mathematical patterns or dead-ends discovered that are highly useful for the Boss Task]
'''



# ==========================================
# 2. LOAD DATASET FROM HUGGING FACE
# ==========================================
print(f"Downloading dataset directly from {DATASET_HUB_NAME}...")
dataset = load_dataset(DATASET_HUB_NAME, split="train")

# ==========================================
# 3. LOAD MODEL & TOKENIZER
# ==========================================
max_seq_length = 4096 
dtype = torch.bfloat16 # A100 supports bfloat16 natively

print("Loading Unsloth FastLanguageModel...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True,
)

# Apply the official Llama-3.1 chat template logic to the tokenizer
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# ==========================================
# 4. CUSTOM FORMATTER FUNCTION
# ==========================================

def custom_formatting_func(examples):
    """
    Custom formatter to map the formatted dataset columns into the
    Llama-3.1 Instruct chat template format.

    The formatted dataset contains:
    - 'input': User prompt formatted with magic_formatter (includes main_task, boss_task, prompt, fragment)
    - 'output': Expected model response
    """
    texts = []

    for i in range(len(examples["input"])):
        user_text = examples["input"][i]
        assistant_text = examples["output"][i]

        # Build the conversation payload
        messages = [
            {"role": "system", "content": abyme_system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]

        # Apply the Llama-3.1 template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(formatted_text)

    return { "text" : texts }

print("Formatting dataset with custom mapper and Llama-3.1 chat template...")
dataset = dataset.map(custom_formatting_func, batched = True, remove_columns = ["input", "output"])

# ==========================================
# 5. INITIALIZE LoRA (PEFT)
# ==========================================
print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 128,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# ==========================================
# 6. SETUP TRAINER & START SFT
# ==========================================
print("Configuring SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4, 
        warmup_steps = 10,
        num_train_epochs = 2,
        learning_rate = 3e-5,
        fp16 = False,
        bf16 = True, 
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = True,
    ),
)

print("Starting training...")
trainer_stats = trainer.train()

# ==========================================
# 7. PUSH MODEL TO HUGGING FACE
# ==========================================
print(f"Training complete! Pushing LoRA adapters to {MODEL_HUB_NAME}...")

model.push_to_hub(MODEL_HUB_NAME, token = HF_TOKEN)
tokenizer.push_to_hub(MODEL_HUB_NAME, token = HF_TOKEN)

print("All tasks completed successfully!")