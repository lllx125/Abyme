"""
Data Generation for Iterative Training

This module handles data management, generation, rating, and testing for the
iterative training loop on the MATH dataset.
"""

import json
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

from benchmark.math_full_benchmark import MATHFullBenchmark
from benchmark.base import extract_boxed_answer
from abyme.batch_runner import ParallelTreeOrchestrator
from abyme.model import Model
from training.rate_trace import rate_all
from abyme.vllm_model import LocalVLLMModel

class DataManager(MATHFullBenchmark):
    """
    Data manager class for iterative MATH training.

    Handles:
    - Selecting training/test subsets from each level
    - Generating model outputs with parallelization
    - Rating outputs for KTO (good/bad)
    - Testing and scoring on test sets
    """

    def __init__(
        self,
        samples: Tuple[int, int, int, int, int],
        tests: Tuple[int, int, int, int, int],
        iteration: int,
        num_gen_per_question: int = 1,
        seed: int = 42
    ):
        """
        Initialize DataManager.

        Args:
            samples: 5-tuple of number of training samples per level (level 1-5)
            tests: 5-tuple of number of test samples per level (level 1-5)
            iteration: Iteration number (used in file naming)
            num_gen_per_question: Number of times to run generation on each question
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.samples = samples
        self.tests = tests
        self.iteration = iteration
        self.num_gen_per_question = num_gen_per_question
        self.seed = seed

        # Store selected data
        self.train_data: List[Dict[str, Any]] = []
        self.test_data: List[Dict[str, Any]] = []
        
        # paths
        self.train_data_path = Path(f"data/{self.iteration}.jsonl")
        self.train_output_path = Path(f"results/{self.iteration}_results.jsonl")
        self.rated_output_path = Path(f"results/{self.iteration}_rated.jsonl")
        self.test_output_path = Path(f"results/{self.iteration}_test_results.jsonl")
        self.test_scored_output_path = Path(f"results/{self.iteration}_test_results_scored.jsonl")
        
        # Huggingface repo IDs
        self.hf_repo_raw = f"Lixing-Li/Abyme-Training-Dataset-Raw-Iteration-{self.iteration}"
        self.hf_repo_rated = f"Lixing-Li/Abyme-Training-Dataset-Rated-Iteration-{self.iteration}"
        self.hf_repo_test_scored = f"Lixing-Li/Abyme-Training-Dataset-Test-Scored-Iteration-{self.iteration}"
        self.hf_repo_trained_model = f"Lixing-Li/Abyme-Trained-Iteration-{self.iteration}"
        
                # Load full dataset
        self._load_and_select_data()

    def _load_and_select_data(self):
        """
        Load full MATH dataset and select training/test subsets.

        For each level, uniformly randomly select problems from each type
        according to samples[level-1] and tests[level-1].

        If saved selection file exists, load from it instead of reselecting.
        """
        # Check if saved selection file exists
        saved_path = self.train_data_path
        if saved_path.exists():
            print(f"\nLoading previously selected data from {saved_path}")
            with saved_path.open('r') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        if item.get('selected_split') == 'train':
                            self.train_data.append(item)
                        elif item.get('selected_split') == 'test':
                            self.test_data.append(item)

            print(f"Loaded {len(self.train_data)} training samples")
            print(f"Loaded {len(self.test_data)} test samples")
            return

        # Otherwise, perform selection
        data_path = Path("data/math_full.jsonl")

        if not data_path.exists():
            self.download()

        # Load all data
        all_data = []
        with data_path.open('r') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line.strip()))

        # Group by (level, type, training)
        grouped_data = defaultdict(list)
        for item in all_data:
            level = item['level_num']
            subject = item['type']
            is_train = item['training']
            key = (level, subject, is_train)
            grouped_data[key].append(item)

        # Set random seed for reproducibility
        random.seed(self.seed)

        # Select training data
        print(f"\nSelecting training data for iteration: {self.iteration}")
        for level in range(1, 6):
            n_samples = self.samples[level - 1]
            if n_samples == 0:
                continue

            # Get all subjects for this level
            subjects = set(k[1] for k in grouped_data.keys() if k[0] == level and k[2])

            # Calculate samples per subject (distribute uniformly)
            samples_per_subject = max(1, n_samples // len(subjects))

            level_samples = []
            for subject in sorted(subjects):
                key = (level, subject, True)  # training=True
                available = grouped_data[key]

                # Sample uniformly
                n_to_sample = min(samples_per_subject, len(available))
                selected = random.sample(available, n_to_sample)
                level_samples.extend(selected)

            # If we need more samples to reach n_samples, sample more
            if len(level_samples) < n_samples:
                all_level_train = [
                    item for key, items in grouped_data.items()
                    if key[0] == level and key[2]
                    for item in items
                    if item not in level_samples
                ]
                additional = random.sample(
                    all_level_train,
                    min(n_samples - len(level_samples), len(all_level_train))
                )
                level_samples.extend(additional)

            # Trim if we oversampled
            level_samples = level_samples[:n_samples]
            self.train_data.extend(level_samples)

            print(f"  Level {level}: Selected {len(level_samples)} training samples")

        # Select test data
        print(f"\nSelecting test data for iteration: {self.iteration}")
        for level in range(1, 6):
            n_tests = self.tests[level - 1]
            if n_tests == 0:
                continue

            # Get all subjects for this level
            subjects = set(k[1] for k in grouped_data.keys() if k[0] == level and not k[2])

            # Calculate tests per subject (distribute uniformly)
            tests_per_subject = max(1, n_tests // len(subjects))

            level_tests = []
            for subject in sorted(subjects):
                key = (level, subject, False)  # training=False
                available = grouped_data[key]

                # Sample uniformly
                n_to_sample = min(tests_per_subject, len(available))
                selected = random.sample(available, n_to_sample)
                level_tests.extend(selected)

            # If we need more tests to reach n_tests, sample more
            if len(level_tests) < n_tests:
                all_level_test = [
                    item for key, items in grouped_data.items()
                    if key[0] == level and not key[2]
                    for item in items
                    if item not in level_tests
                ]
                additional = random.sample(
                    all_level_test,
                    min(n_tests - len(level_tests), len(all_level_test))
                )
                level_tests.extend(additional)

            # Trim if we oversampled
            level_tests = level_tests[:n_tests]
            self.test_data.extend(level_tests)

            print(f"  Level {level}: Selected {len(level_tests)} test samples")

        print(f"\nTotal training samples: {len(self.train_data)}")
        print(f"Total test samples: {len(self.test_data)}")

        # Save selected data to file
        output_path = Path(f"data/{self.iteration}.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving selected data to {output_path}")
        with output_path.open('w') as f:
            # Save all training data
            for item in self.train_data:
                item['selected_split'] = 'train'
                f.write(json.dumps(item) + '\n')

            # Save all test data
            for item in self.test_data:
                item['selected_split'] = 'test'
                f.write(json.dumps(item) + '\n')

        print(f"Selected data saved successfully!")

    def generate_all(
        self,
        model: LocalVLLMModel,
        **recursive_kwargs
    ):
        """
        Generate model outputs for training data using ParallelTreeOrchestrator.

        Generates num_gen_per_question outputs for each training question,
        saves to results/{iteration}_results.jsonl, and optionally uploads to HF.

        Args:
            model: The model to use for generation
            max_concurrent_trees: Number of concurrent trees for ParallelTreeOrchestrator
            upload_to_hf: Whether to upload results to HuggingFace
            hf_repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
            **recursive_kwargs: Additional arguments for RecursiveModel
        """
        output_path = self.train_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"GENERATION: {self.iteration}")
        print(f"{'='*60}")
        print(f"Training samples: {len(self.train_data)}")
        print(f"Generations per question: {self.num_gen_per_question}")
        print(f"Total generations: {len(self.train_data) * self.num_gen_per_question}")
        print(f"Output path: {output_path}")

        # Create orchestrator
        orchestrator = ParallelTreeOrchestrator(
            base_model=model,
            output_jsonl_path=str(output_path),
            **recursive_kwargs
        )

        # Prepare prompts (repeat each question num_gen_per_question times)
        prompts = []
        problem_indices = []
        for idx, item in enumerate(self.train_data):
            for gen_idx in range(self.num_gen_per_question):
                prompts.append(item['problem'])
                problem_indices.append((idx, gen_idx))

        # Run batch generation
        results = orchestrator.run_batch(prompts)
        model.shutdown()

        # Augment results with original problem data, matching by result index
        augmented_results = []
        for result in results:
            prob_idx, gen_idx = problem_indices[result['index']]
            original_data = self.train_data[prob_idx]

            augmented = {
                **result,
                'problem_index': prob_idx,
                'generation_index': gen_idx,
                'level_num': original_data['level_num'],
                'type': original_data['type'],
                'ground_truth': original_data.get('ground_truth', ''),
                'original_problem': original_data.get('problem', '')
            }
            augmented_results.append(augmented)

        # Re-save with augmented data sorted by problem index
        augmented_results.sort(key=lambda r: (r['problem_index'], r['generation_index']))
        with output_path.open('w') as f:
            for result in augmented_results:
                f.write(json.dumps(result) + '\n')

        print(f"\nGeneration complete! Results saved to {output_path} and uploaded to HuggingFace")
        self._upload_to_huggingface(output_path, repo_id=self.hf_repo_raw)

    def rate_all(
        self,
    ):
        """
        Rate generated outputs as good/bad for KTO training.

        Reads from results/{iteration}_results.jsonl and creates
        results/{iteration}_rated.jsonl with 'kto_label' field.
        
        Discard models with medium ratings

        """
        input_path = self.train_output_path
        output_path = self.rated_output_path

        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")

        rate_all(input_path, output_path, self.score)
        
        print(f"Results saved to {output_path} and uploaded to HuggingFace")
        self._upload_to_huggingface(output_path, repo_id=self.hf_repo_rated)

    def test_all(
        self,
        model: LocalVLLMModel,
        **recursive_kwargs
    ):
        """
        Run model on test set and store results.

        Generates outputs for test_data and saves to
        results/{iteration}_test_results.jsonl.

        Args:
            model: The model to use for testing
            max_concurrent_trees: Number of concurrent trees
            **recursive_kwargs: Additional arguments for RecursiveModel
        """
        output_path = self.test_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"TESTING: {self.iteration}")
        print(f"{'='*60}")
        print(f"Test samples: {len(self.test_data)}")
        print(f"Output path: {output_path}")

        # Create orchestrator
        orchestrator = ParallelTreeOrchestrator(
            base_model=model,
            output_jsonl_path=str(output_path),
            **recursive_kwargs
        )

        # Prepare prompts
        prompts = [item['problem'] for item in self.test_data]

        # Run batch testing
        results = orchestrator.run_batch(prompts)
        model.shutdown()

        # Augment results with original problem data, matching by result index
        augmented_results = []
        for result in results:
            i = result['index']
            original_data = self.test_data[i]

            augmented = {
                **result,
                'problem_index': i,
                'level_num': original_data['level_num'],
                'type': original_data['type'],
                'ground_truth': original_data.get('ground_truth', ''),
                'original_problem': original_data.get('problem', '')
            }
            augmented_results.append(augmented)

        # Re-save with augmented data sorted by problem index
        augmented_results.sort(key=lambda r: r['problem_index'])
        with output_path.open('w') as f:
            for result in augmented_results:
                f.write(json.dumps(result) + '\n')

        print(f"\nTesting complete! Results saved to {output_path}")
        
    def _upload_to_huggingface(self, file_path: Path, repo_id: str = ""):
        """
        Upload results to HuggingFace Hub using datasets.push_to_hub.

        Args:
            file_path: Path to the JSONL file to upload
            repo_id: HuggingFace repo ID (optional, defaults to Lixing-Li/Abyme-Training-Dataset-Iteration-{n})
        """
        try:
            from datasets import load_dataset
            from huggingface_hub import login
            import os
            import dotenv

            dotenv.load_dotenv()

            print(f"\nUploading to HuggingFace Hub: {repo_id}")
            print("Logging into Hugging Face...")
            login(token=os.getenv("HF_TOKEN", ""), add_to_git_credential=True)

            # Load the JSONL file as a dataset
            print(f"Loading dataset from {file_path}...")
            dataset = load_dataset("json", data_files=str(file_path), split="train")

            # Upload to HuggingFace
            print(f"Pushing to hub...")
            dataset.push_to_hub(repo_id)

            print(f"✓ Upload complete! Dataset available at: https://huggingface.co/datasets/{repo_id}")

        except ImportError as e:
            print(f"Warning: Required libraries not installed. Skipping upload.")
            print(f"Install with: pip install datasets huggingface_hub python-dotenv")
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error uploading to HuggingFace: {e}")

    def _normalize_answer(self, ans: str) -> str:
        """Normalize answer for comparison (inherited from parent)."""
        # Use parent class normalization from MATH500Benchmark
        return super()._normalize_answer(ans)
    
    def score(self, model_output: str, input: Dict[str, Any]) -> float:
        """Score using ground_truth field from MATH full dataset."""
        input_with_answer = {**input, 'answer': input.get('ground_truth', input.get('answer', ''))}
        return super().score(model_output, input_with_answer)

    def score_all(self):
        """Score results/{iteration}_test_results.jsonl using inherited score function."""
        input_path = self.test_output_path
        output_path = self.test_scored_output_path

        if not input_path.exists():
            raise FileNotFoundError(f"Test results file does not exist: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with input_path.open('r') as infile:
            lines = [line for line in infile if line.strip()]

        total_score = 0.0
        with output_path.open('w') as outfile:
            for line in tqdm(lines, desc=f"Score: {self.iteration}_test_results"):
                json_data = json.loads(line.strip())
                if 'output' not in json_data:
                    raise ValueError(f"Missing 'output' field in {input_path}")
                s = self.score(json_data['output'], json_data)
                total_score += s
                outfile.write(json.dumps({**json_data, 'score': s}) + '\n')

        avg = total_score / len(lines) if lines else 0.0
        print(f"Average score: {avg:.4f} ({int(total_score)}/{len(lines)} correct)")
        print(f"Scored results saved to {output_path} and uploaded to HuggingFace")
        self._upload_to_huggingface(output_path, repo_id=self.hf_repo_test_scored)
        return avg, [json.loads(l)['score'] for l in output_path.open()]

    def check_scores_by_level(self) -> Tuple[float, float, float, float, float, float]:
        """Print average score per level and total average from scored output file."""
        test_name = f"iteration_{self.iteration}"
        results = self._check_scores_from_path(self.test_scored_output_path, test_name, level_field="level_num")
        return results


if __name__ == "__main__":
    base_model = "Lixing-Li/Abyme-Qwen3.5-9B-Test-KTO"
    iteration = 0
    sample = (10,10,10,10,10) 
    test = (10,10,10,10,10)
    data_manager = DataManager(iteration=iteration, samples=sample, tests=test, num_gen_per_question=2)
    model = LocalVLLMModel(model_path=base_model)
    print("Model loaded successfully!")
    data_manager.test_all(model=model, max_depth=5, max_parallel_workers=5, max_call=50, max_chain_length=5)
    data_manager.score_all()
    data_manager.check_scores_by_level()
    # data_manager.generate_all(model=model, max_concurrent_trees=10, max_depth=5, max_parallel_workers=5, max_call=50, max_chain_length=5)
    # data_manager.rate_all()