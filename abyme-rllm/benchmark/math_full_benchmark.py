"""
MATH Full Dataset Benchmark

Purpose: Full MATH dataset benchmark implementation (12,500 problems) as a child class of Benchmark.
"""

import json
from pathlib import Path
from benchmark.math500_benchmark import MATH500Benchmark
from benchmark.base import extract_boxed_answer


class MATHFullBenchmark(MATH500Benchmark):
    """Full MATH dataset benchmark for evaluating math problem solving (12,500 problems)."""

    @property
    def name(self):
        """The name of the benchmark."""
        return "math_full"

    def download(self):
        """
        Download and save full MATH dataset from Hugging Face.
        Save the dataset to "data/math_full.jsonl" for use in generation and scoring.

        Filters out invalid level problems.
        Splits each level+type combination into 70% training and 30% testing.
        Sorts data by level, then training/testing, then type (subject).
        Extracts boxed answer and stores as ground_truth field.
        """
        from datasets import load_dataset
        from collections import defaultdict

        output_path = Path(f"data/math_full.jsonl")

        print("Downloading the full MATH dataset...")

        # Load all subject categories individually
        subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus"
        ]

        # Combine all subjects (ignore original train/test split)
        all_problems = []

        for subject in subjects:
            print(f"Loading {subject}...")
            dataset = load_dataset("EleutherAI/hendrycks_math", subject)

            # Add all problems from both splits
            for item in dataset["train"]:
                problem_dict = dict(item)
                all_problems.append(problem_dict)

            for item in dataset["test"]:
                problem_dict = dict(item)
                all_problems.append(problem_dict)

        print(f"\nTotal problems loaded: {len(all_problems)}")

        # Filter out invalid level problems
        def extract_level_num(level_str):
            """Extract numeric level (1-5) or None if invalid."""
            if isinstance(level_str, str) and "Level" in level_str:
                try:
                    parsed = int(level_str.split()[-1])
                    if 1 <= parsed <= 5:
                        return parsed
                except (ValueError, IndexError):
                    pass
            return None

        valid_problems = []
        invalid_count = 0
        for problem in all_problems:
            level_num = extract_level_num(problem.get("level"))
            if level_num is not None:
                problem["level_num"] = level_num
                valid_problems.append(problem)
            else:
                invalid_count += 1

        print(f"Removed {invalid_count} problems with invalid levels")
        print(f"Valid problems remaining: {len(valid_problems)}")

        # Group by (level, type) for 70/30 split
        level_type_groups = defaultdict(list)
        for problem in valid_problems:
            key = (problem["level_num"], problem["type"])
            level_type_groups[key].append(problem)

        # Apply 70/30 split for each group
        all_split_problems = []
        for (level, subject), problems in level_type_groups.items():
            # Sort for consistency
            problems.sort(key=lambda x: x.get("problem", ""))

            split_idx = int(len(problems) * 0.7)

            # First 70% are training
            for i, problem in enumerate(problems):
                if i < split_idx:
                    problem["training"] = True
                else:
                    problem["training"] = False
                all_split_problems.append(problem)

        print(f"\nApplied 70/30 training/testing split for each level+type combination")

        # Statistics by level and type
        stats_by_level = defaultdict(lambda: {"total": 0, "training": 0, "testing": 0})
        stats_by_level_type = defaultdict(lambda: {"total": 0, "training": 0, "testing": 0})

        for problem in all_split_problems:
            level = problem["level_num"]
            subject = problem["type"]
            is_training = problem["training"]

            # Level stats
            stats_by_level[level]["total"] += 1
            if is_training:
                stats_by_level[level]["training"] += 1
            else:
                stats_by_level[level]["testing"] += 1

            # Level+Type stats
            key = (level, subject)
            stats_by_level_type[key]["total"] += 1
            if is_training:
                stats_by_level_type[key]["training"] += 1
            else:
                stats_by_level_type[key]["testing"] += 1

        # Print statistics
        print("\n" + "="*60)
        print("STATISTICS BY LEVEL")
        print("="*60)
        for level in sorted(stats_by_level.keys()):
            stats = stats_by_level[level]
            print(f"Level {level}: {stats['total']} total ({stats['training']} train, {stats['testing']} test)")

        print("\n" + "="*60)
        print("STATISTICS BY LEVEL AND TYPE")
        print("="*60)
        for level in sorted(set(k[0] for k in stats_by_level_type.keys())):
            print(f"\nLevel {level}:")
            level_total = 0
            # Get all subjects for this level
            level_subjects = sorted(set(k[1] for k in stats_by_level_type.keys() if k[0] == level))
            for subject in level_subjects:
                key = (level, subject)
                stats = stats_by_level_type[key]
                level_total += stats["total"]
                print(f"  {subject:30s}: {stats['total']:4d} total ({stats['training']:4d} train, {stats['testing']:4d} test)")
            print(f"  {'Total for Level ' + str(level):30s}: {level_total:4d}")

        total_training = sum(stats_by_level[l]["training"] for l in stats_by_level)
        total_testing = sum(stats_by_level[l]["testing"] for l in stats_by_level)
        print("\n" + "="*60)
        print(f"OVERALL TOTAL: {len(all_split_problems)} ({total_training} train, {total_testing} test)")
        print("="*60)

        # Sort by level, then training/testing (testing first), then type (subject)
        def sort_key(item):
            level_num = item["level_num"]
            training = item["training"]
            subject = item["type"]
            return (level_num, training, subject)

        all_split_problems.sort(key=sort_key)

        print("\nData sorted by: level → training/testing → type (subject)")

        # Inspect the first problem after sorting
        sample = all_split_problems[0]
        print("\n--- Sample Problem (First after sorting) ---")
        print(f"Subject: {sample['type']}")
        print(f"Difficulty: {sample['level']}")
        print(f"Training: {sample['training']}")
        print(f"\nProblem:\n{sample['problem'][:200]}...")
        print(f"\nSolution:\n{sample['solution'][:200]}...")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save all problems to JSONL file
        print(f"\nSaving all problems to {output_path}...")
        with output_path.open("w", encoding="utf-8") as f:
            for item in all_split_problems:
                # Build output dictionary
                output_dict = {}
                for key in item.keys():
                    output_dict[key] = item[key]

                # Extract ground truth from solution using extract_boxed_answer
                solution = output_dict.get("solution", "")
                ground_truth = extract_boxed_answer(solution)
                output_dict["ground_truth"] = ground_truth

                # Get the problem text
                problem_text = output_dict.get("problem") or output_dict.get("question") or ""

                # Add instruction to put answer in \boxed{} format
                problem_with_instruction = (
                    f"{problem_text}\n\n"
                    f"Please provide your final answer in the format \\boxed{{answer}}."
                )

                # Update the problem field with the instruction
                output_dict["problem"] = problem_with_instruction

                f.write(json.dumps(output_dict) + "\n")

        print("Done!")
        print(f"Download and save complete! Saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    benchmark = MATHFullBenchmark()
    benchmark.download()
