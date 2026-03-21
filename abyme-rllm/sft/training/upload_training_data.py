import os
import dotenv
from datasets import load_dataset
from huggingface_hub import login
from abyme.magic import magic_formatter

dotenv.load_dotenv()

# Path to the JSONL file
DATA_PATH = "data/sft_training_data.jsonl"

def upload_to_huggingface():
    # 1. Authenticate with Hugging Face
    print("Logging into Hugging Face...")
    login(token=os.getenv("HF_TOKEN",""), add_to_git_credential=True)

    # 2. Load the local JSONL file
    file_path = DATA_PATH
    print(f"Loading dataset from {file_path}...")
    dataset = load_dataset("json", data_files=file_path, split="train")
    print(f"Success! Dataset loaded with {len(dataset)} samples.")

    # 3. Upload RAW dataset (unformatted)
    raw_repo_id = "Lixing-Li/Abyme-finetune-dataset-raw"
    print(f"\nUploading RAW dataset to https://huggingface.co/datasets/{raw_repo_id}...")
    dataset.push_to_hub(raw_repo_id)
    print("✓ Raw dataset upload complete!")

    # 4. Create FORMATTED dataset with magic_formatter
    print("\nApplying magic_formatter to create formatted dataset...")

    def apply_magic_formatter(example):
        """Apply magic_formatter to create the 'input' field"""
        if not example['prompt']:
            raise ValueError(f"Missing 'prompt' in example: {example}")
        formatted_input = magic_formatter(
            prompt=example['prompt'] or 'None',
            main_problem=example['main_task'] or 'None',
            boss_problem=example['boss_task'] or 'None',
            fragment=example['fragment'] or 'None'
        )
        return {
            'input': formatted_input,
            'output': example['output']
        }

    formatted_dataset = dataset.map(apply_magic_formatter).select_columns(["input", "output"])
    print(f"✓ Formatted dataset created with {len(formatted_dataset)} samples.")

    # 4.5. Calculate and print max character count
    print("\nCalculating maximum character count for input + output...")
    max_chars = 0
    max_example_idx = -1
    max_input_chars = 0
    max_output_chars = 0

    for idx, example in enumerate(formatted_dataset):
        input_len = len(example['input']) if example['input'] else 0
        output_len = len(example['output']) if example['output'] else 0
        combined_len = input_len + output_len

        if combined_len > max_chars:
            max_chars = combined_len
            max_example_idx = idx
            max_input_chars = input_len
            max_output_chars = output_len

    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print(f"Maximum combined length: {max_chars:,} characters")
    print(f"  - Input length:  {max_input_chars:,} characters")
    print(f"  - Output length: {max_output_chars:,} characters")
    print(f"  - Example index: {max_example_idx}")
    print(f"  - Estimated tokens (÷4): ~{max_chars//4:,} tokens")
    print("="*60 + "\n")

    # 5. Upload FORMATTED dataset
    formatted_repo_id = "Lixing-Li/Abyme-finetune-dataset-formatted"
    print(f"\nUploading FORMATTED dataset to https://huggingface.co/datasets/{formatted_repo_id}...")
    formatted_dataset.push_to_hub(formatted_repo_id)
    print("✓ Formatted dataset upload complete!")

    print("\n" + "="*60)
    print("✅ UPLOAD SUMMARY")
    print("="*60)
    print(f"Raw dataset:       {raw_repo_id}")
    print(f"Formatted dataset: {formatted_repo_id}")
    print(f"Total samples:     {len(dataset)}")
    print("="*60)

if __name__ == "__main__":
    upload_to_huggingface()
