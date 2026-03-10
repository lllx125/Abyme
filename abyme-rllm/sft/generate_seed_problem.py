import os
import json
import re
from typing import List, Tuple
from dotenv import load_dotenv
from abyme.model import deepseek

# Load environment variables from .env file
load_dotenv()

OUTPUT_FILE = "data/seed_problems.jsonl"


def generate_seed_problems(problems_per_category: int = 20, overwrite: bool = True, max_retries: int = 3, categories: List[str] = []) -> List[Tuple[str, str]]:
    """
    Generate a diverse set of seed problems using DeepSeek API.

    This function calls DeepSeek to dynamically generate problems across various categories,
    saves them to seed_problems.jsonl, and returns them as a list.

    Args:
        problems_per_category: Number of problems to generate for each category
        overwrite: If True, overwrite existing file. If False, append to it. Default is True.
        max_retries: Maximum number of retry attempts for failed API calls. Default is 3.

    Returns:
        List of (category, problem) tuples
    """


    # Prompt template for generating problems
    prompt_template = """Generate {n} diverse and challenging {category} problems suitable for training a recursive reasoning model.

Requirements:
1. Each problem should be self-contained and clearly stated
2. Problems should vary in difficulty from medium to hard
3. Problems should require step-by-step reasoning to solve
4. Avoid trivial or overly simple problems
5. The problem should encourage the use of recursive problem-solving techniques, such as breaking down into sub-problems
6. The answer should be able to verify and stated as text
7. Return ONLY a valid JSON array of problem strings, with no additional text

Format your response as a JSON array:
["problem 1 text", "problem 2 text", "problem 3 text", ...]

Category: {category}
Number of problems: {n}"""

    all_problems = []
    output_file = OUTPUT_FILE

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)

    print(f"Generating seed problems across {len(categories)} categories...")
    print(f"Problems per category: {problems_per_category}")
    print(f"Output file: {output_file}")
    print(f"Mode: {'Overwrite' if overwrite else 'Append'}")
    print()

    # Clear file if overwrite mode is enabled
    if overwrite:
        with open(output_file, 'w') as f:
            pass  # Just clear the file

    failed_categories = []

    # Initialize DeepSeek model once for all categories
    model = deepseek(
        reasoning=False,
        system_prompt="You are a helpful assistant that generates educational problems. Always respond with valid JSON."
    )

    for category_idx, category in enumerate(categories):
        print(f"[{category_idx + 1}/{len(categories)}] Generating {category} problems...")

        success = False
        for attempt in range(max_retries):
            try:
                # Format the prompt for this category
                prompt = prompt_template.format(
                    n=problems_per_category,
                    category=category.replace('_', ' ')
                )

                # Call DeepSeek API using the model
                content = model.generate(prompt, max_attempt=1)
                if not content:
                    print(f"  Attempt {attempt + 1}/{max_retries}: Empty response from API.")
                    if attempt < max_retries - 1:
                        print(f"  Retrying...")
                        continue
                    else:
                        print(f"  Skipping category after {max_retries} attempts.")
                        break

                content = content.strip()

                # Try to extract JSON if there's additional text
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

                # Parse JSON array of problems
                problems = json.loads(content)

                if not isinstance(problems, list):
                    print(f"  Attempt {attempt + 1}/{max_retries}: Expected list, got {type(problems)}.")
                    if attempt < max_retries - 1:
                        print(f"  Retrying...")
                        continue
                    else:
                        print(f"  Skipping category after {max_retries} attempts.")
                        break

                # Validate and save each problem to JSONL immediately after receiving response
                valid_problems_count = 0
                with open(output_file, 'a') as f:
                    for problem in problems:
                        if isinstance(problem, str) and problem.strip():
                            problem_data = {
                                "category": category,
                                "problem": problem.strip()
                            }

                            # Validate that the JSON can be serialized correctly
                            try:
                                json_str = json.dumps(problem_data, ensure_ascii=False)
                                # Validate it can be parsed back
                                json.loads(json_str)
                                # If validation passes, write to file
                                f.write(json_str + '\n')
                                all_problems.append((category, problem.strip()))
                                valid_problems_count += 1
                            except (TypeError, ValueError, json.JSONDecodeError) as e:
                                print(f"  Warning: Skipping invalid problem due to JSON error: {e}")
                                print(f"  Problem preview: {problem[:100]}...")
                                continue

                print(f"  Generated {len(problems)} problems, {valid_problems_count} valid for {category}")
                print(f"  Saved to {output_file}")
                success = True
                break  # Success, exit retry loop

            except json.JSONDecodeError as e:
                print(f"  Attempt {attempt + 1}/{max_retries}: Error parsing JSON for {category}: {e}")
                if 'content' in locals() and content:
                    print(f"  Response content: {content[:200]}...")
                if attempt < max_retries - 1:
                    print(f"  Retrying...")
                    continue
            except Exception as e:
                print(f"  Attempt {attempt + 1}/{max_retries}: Error generating problems for {category}: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying...")
                    continue

        if not success:
            print(f"  FAILED: Could not generate valid problems for category '{category}' after {max_retries} attempts.")
            failed_categories.append(category)

        print()

    print(f"Successfully generated {len(all_problems)} total seed problems")
    print(f"Saved to: {output_file}")

    if failed_categories:
        print(f"\nFailed categories ({len(failed_categories)}):")
        for cat in failed_categories:
            print(f"  - {cat}")
    else:
        print(f"\nAll categories succeeded!")

    print()

    return all_problems


if __name__ == "__main__":
    categories = [
        "Algebraic Equation Solving", "Differential Calculus", "Integral Calculus",
        "Number Theory", "Geometric Proofs", "Combinatorics", "Probability Calculation",
        "Graph Theory", "Linear Algebra", "Set Theory", "Boolean Algebra",
        "Propositional Logic", "Predicate Logic", "Game Theory", "Cryptographic Ciphers",
        "Sorting Algorithms", "Search Algorithms", "Dynamic Programming",
        "Recursive Function Tracing", "Big O Analysis", "Regular Expressions",
        "SQL Query Construction", "Database Normalization", "Bitwise Operations",
        "Network Subnetting", "Chemical Equation Balancing", "Stoichiometry", "Physics Kinematics",
        "Circuit Analysis", "Thermodynamics", "Orbital Mechanics", "Genetics Punnett Squares",
        "DNA Sequence Alignment", "Organic Chemistry Naming", "Chess Endgame Tactics",
        "Sudoku Solving", "Logic Grid Puzzles", "Cryptarithmetic", "Einstein Riddles",
        "Syllogism Validation", "Truth Table Construction", "Financial Compound Interest",
        "Accounting Balance Sheets", "Supply Chain Optimization", "Taxonomy Classification",
        "Syntax Tree Parsing", "Phonetic Transcription", "Music Theory Analysis",
        "Counterpoint Rules", "Assembly Debugging"
    ]
    categories = [
        "Ethical Dilemma Resolution", "Regulatory Compliance Auditing", "Patent Infringement Assessment",
        "Supply Chain Logistics Planning", "Project Management Critical Path", 
        "Crisis Communication Sequencing", "Negotiation BATNA Calculation", "Market Segmentation Logic", 
        "Investment Portfolio Rebalancing", "Business Risk Assessment Matrix", "Corporate Process Mapping", 
        "Root Cause Analysis (Business)", "Historical Counterfactual Simulation", "Geopolitical Strategy Gaming",
        "Diplomatic Protocol Sequencing", "Genealogical Lineage Verification", "Archaeological Stratigraphy Logic",
        "Linguistic Syntax Tree Generation", "Translation Ambiguity Resolution", "Phonetic Transcription Logic",
        "Rhetorical Structure Analysis", "Logical Fallacy Identification", "Syllogism Validity Checking",
        "Philosophical Argument Reconstruction", "Debate Rebuttal Formulation", "Narrative Plot Hole Detection",
        "Screenplay Beat Sheet Structuring", "Fictional World-Building Consistency", "Poetic Meter Scansion",
        "Music Theory Harmony Analysis", "Counterpoint Composition Rules", "Color Theory Palette Generation", 
        "Interior Design Space Planning", "Fashion Style Taxonomy", "Game Theory Payoff Matrix", "Board Game Rule Adjudication",
        "Chess Move Explanation", "Poker Hand Probability Analysis", "RPG Character Stat Optimization", "Sports Playbook Logic",
        "Travel Itinerary Optimization", "Event Seating Arrangement Logic", "Ingredient Substitution Logic",
        "Dietary Restriction Menu Planning", "Personal Budget Allocation", "Tax Deduction Categorization",
        "Library Classification Logic", "Journalistic Fact-Checking", "Cryptic Crossword Solving", "Riddle Deconstruction"
    ]
    generate_seed_problems(categories=categories, overwrite=False, problems_per_category=10, max_retries=3)