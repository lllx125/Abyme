"""
MATH-100 Benchmark

Purpose: MATH-100 dataset benchmark — 20 randomly sampled questions per difficulty
level (5 levels = 100 total). Subclass of MATH500Benchmark that overrides
generate_all and score_all with parallel ThreadPoolExecutor for vLLM throughput,
and inherits score() from MATH500Benchmark.
"""

import json
from typing import Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from abyme.batch_runner import ParallelTreeOrchestrator
from abyme.vllm_model import LocalVLLMModel
from benchmark.math500_benchmark import MATH500Benchmark


class MATH100Benchmark(MATH500Benchmark):
    """
    MATH-100: 20 randomly sampled questions per difficulty level (5 levels = 100 total).
    Overrides generate_all with ParallelTreeOrchestrator for vLLM throughput,
    and score_all with parallel ThreadPoolExecutor. Inherits score() from MATH500Benchmark.
    """

    @property
    def name(self):
        return "math100"

    def download(self):
        """
        Download MATH-500 and sample 20 questions per difficulty level to form MATH-100.
        Saves the dataset to "data/math100.jsonl".
        """
        import random
        from datasets import load_dataset

        output_path = Path(f"data/{self.name}.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("Downloading MATH-500 dataset for MATH-100 sampling...")
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

        # Group items by difficulty level
        by_level: Dict[Any, list] = {}
        for item in dataset:  # type: ignore
            level = item.get("level", "Unknown")  # type: ignore
            by_level.setdefault(level, []).append(dict(item))  # type: ignore

        selected = []
        for level in sorted(by_level.keys()):
            items = by_level[level]
            k = min(20, len(items))
            sampled = random.sample(items, k)
            selected.extend(sampled)
            print(f"  Level {level}: sampled {k}/{len(items)}")

        random.shuffle(selected)

        with output_path.open("w", encoding="utf-8") as f:
            for item in selected:
                problem_text = item.get("problem") or item.get("question") or ""
                item["problem"] = (
                    f"{problem_text}\n\n"
                    f"Please provide your final answer in the format \\boxed{{answer}}."
                )
                f.write(json.dumps(item) + "\n")

        print(f"MATH-100 saved to {output_path} ({len(selected)} problems)")

    def generate_all(self, model: LocalVLLMModel, test_name: str, **recursive_kwargs):
        """
        Parallel generation using ParallelTreeOrchestrator to saturate the vLLM async engine.

        Args:
            model: An instance of LocalVLLMModel
            test_name: Name for this run (used for output file naming)
            **recursive_kwargs: Arguments passed to ParallelTreeOrchestrator / RecursiveModel
                                 (e.g., max_concurrent_trees, max_depth, max_parallel_workers)
        """
        input_path = Path(f"data/{self.name}.jsonl")
        output_path = Path(f"results/{self.name}/{test_name}.jsonl")

        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with input_path.open("r") as f:
            problems = [json.loads(line) for line in f if line.strip()]

        orchestrator = ParallelTreeOrchestrator(
            base_model=model,
            output_jsonl_path=str(output_path),
            **recursive_kwargs
        )

        prompts = [item["problem"] for item in problems]
        results = orchestrator.run_batch(prompts)
        model.shutdown()

        # Augment results with original problem metadata
        augmented = []
        for result in results:
            idx = result["index"]
            augmented.append({**problems[idx], **result})

        augmented.sort(key=lambda r: r["index"])

        with output_path.open("w") as f:
            for record in augmented:
                f.write(json.dumps(record) + "\n")

        print(f"Generation complete. Saved to {output_path}")

    def score_all(self, test_name: str, max_workers: int = 50):
        """
        Parallel scoring using ThreadPoolExecutor.
        Inherits the single-item score() method from MATH500Benchmark.

        Args:
            test_name: The name of the generation run to score
            max_workers: Number of concurrent scoring threads
        """
        input_path = Path(f"results/{self.name}/{test_name}.jsonl")
        output_path = Path(f"results/{self.name}/{test_name}_scored.jsonl")

        if not input_path.exists():
            raise FileNotFoundError(f"Generated results file does not exist: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with input_path.open("r") as f:
            records = [json.loads(line) for line in f if line.strip()]

        results: list = [None] * len(records)

        def _score(idx_item):
            idx, item = idx_item
            if "output" not in item:
                raise ValueError(f"Missing 'output' field in record {idx}")
            s = self.score(item["output"], item)
            return idx, {**item, "score": s}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_score, (i, r)): i for i, r in enumerate(records)}
            for future in tqdm(as_completed(futures), total=len(records), desc=f"Score: {test_name}"):
                idx, record = future.result()
                results[idx] = record

        with output_path.open("w") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")

        print(f"Scoring complete. Saved to {output_path}")


if __name__ == "__main__":
    MODEL_PATH = "Lixing-Li/Abyme-Qwen3.5-9B-SFT"
    TEST_NAME = "SFT-base"

    benchmark = MATH100Benchmark()

    data_path = Path(f"data/{benchmark.name}.jsonl")
    if not data_path.exists():
        print("MATH-100 dataset not found. Downloading and sampling...")
        benchmark.download()

    model = LocalVLLMModel(model_path=MODEL_PATH)
    print("Model loaded. Starting parallel generation...")

    benchmark.generate_all(
        model,
        test_name=TEST_NAME,
        max_concurrent_trees=20,
        max_depth=5,
        max_parallel_workers=5,
        max_call=50,
        max_chain_length=5,
    )

    benchmark.score_all(test_name=TEST_NAME)
    benchmark.check_scores(test_name=TEST_NAME)
    scores = benchmark.check_scores_by_level(
        Path(f"results/{benchmark.name}/{TEST_NAME}_scored.jsonl"), TEST_NAME
    )
    benchmark.append_score_to_hub(scores, TEST_NAME)
