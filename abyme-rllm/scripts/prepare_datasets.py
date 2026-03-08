#!/usr/bin/env python3
"""
CLI script to download, format, and save datasets for Abyme training.

Usage:
    python scripts/prepare_datasets.py --task sft --output_dir data/processed --limit 1000
    python scripts/prepare_datasets.py --task rl --output_dir data/processed
    python scripts/prepare_datasets.py --task benchmark --output_dir data/processed
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Add parent directory to path to import abyme
sys.path.insert(0, str(Path(__file__).parent.parent))

from abyme.data.loading import DatasetFactory


def save_jsonl(data, output_path):
    """
    Save data to JSONL format.

    Args:
        data: List of dictionaries to save
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(data)} samples to {output_path}")


def print_stats(data):
    """
    Print dataset statistics.

    Args:
        data: List of data samples
    """
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(data)}")

    if data:
        # Count by source
        sources = {}
        for item in data:
            source = item.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        print("  Sources:")
        for source, count in sources.items():
            print(f"    - {source}: {count}")

        # Average lengths
        avg_prompt = sum(len(item['prompt']) for item in data) / len(data)
        avg_solution = sum(len(item['solution']) for item in data) / len(data)

        print(f"  Avg prompt length: {avg_prompt:.0f} chars")
        print(f"  Avg solution length: {avg_solution:.0f} chars")

        # Show first example
        print("\nFirst example:")
        first = data[0]
        print(f"  ID: {first['id']}")
        print(f"  Prompt: {first['prompt'][:100]}...")
        print(f"  Answer: {first['answer']}")
        print(f"  Solution: {first['solution'][:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Abyme training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare SFT data with 1000 samples
  python scripts/prepare_datasets.py --task sft --limit 1000

  # Prepare RL data (all hard problems)
  python scripts/prepare_datasets.py --task rl

  # Prepare AIME benchmark
  python scripts/prepare_datasets.py --task benchmark --output_dir data/eval
        """
    )

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['sft', 'rl', 'benchmark'],
        help='Type of dataset to prepare'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed datasets (default: data/processed)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to load (default: train)'
    )

    parser.add_argument(
        '--levels',
        type=int,
        nargs='+',
        default=[3, 4, 5],
        help='Difficulty levels for RL data (default: 3 4 5)'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Preparing {args.task.upper()} Dataset")
    print(f"{'='*60}\n")

    # Initialize factory
    factory = DatasetFactory()

    # Load data based on task
    try:
        if args.task == 'sft':
            print(f"Task: Supervised Fine-Tuning (SFT)")
            print(f"Source: AI-MO/NuminaMath-CoT")
            print(f"Split: {args.split}")
            if args.limit:
                print(f"Limit: {args.limit} samples")
            print()

            data = factory.load_sft_data(split=args.split, limit=args.limit)
            output_filename = f"sft_{args.split}.jsonl"

        elif args.task == 'rl':
            print(f"Task: Reinforcement Learning (RL)")
            print(f"Source: hendrycks/competition_math")
            print(f"Split: {args.split}")
            print(f"Levels: {args.levels}")
            if args.limit:
                print(f"Limit: {args.limit} samples")
            print()

            data = factory.load_rl_data(
                split=args.split,
                focus_levels=args.levels,
                limit=args.limit
            )
            output_filename = f"rl_{args.split}_levels_{'_'.join(map(str, args.levels))}.jsonl"

        elif args.task == 'benchmark':
            print(f"Task: AIME Benchmark (Evaluation)")
            print(f"Source: AI-MO/aimo-validation-aime (or filtered MATH)")
            if args.limit:
                print(f"Limit: {args.limit} samples")
            print()

            data = factory.load_aime_benchmark(limit=args.limit)
            output_filename = "aime_benchmark.jsonl"

        else:
            print(f"Error: Unknown task '{args.task}'")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print statistics
    print_stats(data)

    # Save to JSONL
    output_path = Path(args.output_dir) / output_filename
    print(f"\nSaving to {output_path}...")

    try:
        save_jsonl(data, output_path)
    except Exception as e:
        print(f"\n❌ Error saving dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*60}")
    print("✓ Dataset preparation complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
