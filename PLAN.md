Day 1: The Controller & Tokenizer

Task: Write the Python Controller script that handles the </run> stop token, parses XML, and manages the recursion stack.

Task: Set up the Tokenizer. Add <elaborate>, </elaborate>, <response>, </response>, </run>.

Test: Load DeepSeek-R1-Distill-Llama-8B in 4-bit (via Unsloth) on your laptop. Manually force it to output these tokens to see if your controller parses them correctly.

Day 2: The Verifier

Task: Write the verify_math(answer, solution) function.

Test: Download the AIME and AMC subsets from the NuminaMath or Hendrycks MATH dataset. Run your verifier against the known solutions to ensure it's 100% accurate.

Day 3: The SFT Dataset Generator (Script)

Task: Write a script to call OpenAI/DeepSeek API. It should take a MATH problem and prompt the model to "Solve this using recursive <elaborate> tags."

Prompt Engineering: Spend time here. If the teacher (GPT-4) produces bad data, your student (8B) will fail.

Day 4: The "Dry Run" Training

Task: Set up the Unsloth SFT training script locally.

Action: Run a training loop on 10 samples on your 4080.

Goal: Ensure no OOM (Out of Memory) errors and that the loss actually decreases.

Phase 2: Data Generation (Days 6-7)
Goal: Create the "Textbook" for your model.
Hardware: API.
Cost: ~$50 - $80.

Day 6: Generate SFT Data

Action: Generate 2,000 High-Quality Examples using GPT-4o (or DeepSeek-Chat V3, which is cheaper and excellent at reasoning).

Optimization: Do not generate data for GRPO. GRPO only needs the Question and the Final Answer (which you already have from the open-source dataset). You only need generated traces for the SFT phase.

Day 7: Data Cleaning

Action: Write a regex script to filter out broken XML tags from the generated data. If a sample is malformed, discard it. Quality > Quantity.

Phase 3: Supervised Fine-Tuning (Days 8-9)
Goal: Teach the model how to speak your language (Syntax).
Hardware: Cloud H100 (Lambda/RunPod).
Cost: ~$10 - $20.

Day 8: The SFT Run

Rent: 1x H100 (80GB).

Job: Train for 2-3 epochs on your 2k dataset.

Time: Should take < 2 hours on an H100.

While Waiting: Draft the "Methodology" section of your paper. Describe the architecture and the controller logic.

Day 9: Local Evaluation (The "Sanity Check")

Download: Pull the LoRA adapters back to your laptop.

Test: Run the model on 50 unseen problems.

Check: Does it actually stop at </run>? Does it use the tags? If yes, proceed. If no, debug and re-train (Phase 3).

Phase 4: GRPO - The "Main Event" (Days 10-14)
Goal: Teach the model when to recurse to get the right answer (Reasoning).
Hardware: Cloud H100.
Cost: ~$150 - $300 (The bulk of your budget).

Day 10: GRPO Setup & Dry Run

Config: Set up the GRPO trainer.

Hyperparams: Group size = 8 or 16. (This means for every question, it generates 16 attempts).

Local Test: Run 1 step on your 4080 just to check the code doesn't crash.

Day 11-12: The Training Run

Rent: 1x H100 (or 2x if you want speed).

Job: Run GRPO on the AIME/AMC dataset (hard math).

Time: This will likely take 12-24 hours depending on convergence.

Optimization (While Waiting):

Write the "Related Works" section. Compare yourself to RLM and ReCAP.

Build the Evaluation Suite: Write the script that will automatically benchmark the final model.

Day 13: Checkpoint Analysis

Monitor the "Reward" curve in WandB. If it's flatlining, kill the run (save money). If it's going up, let it ride.

Day 14: Final Model Consolidation

Merge the LoRA adapters into the base model (if verification passes) or keep them separate. Download the final weights.

Phase 5: Evaluation & Paper (Days 15-28)
Goal: Prove you won.
Hardware: Local (Inference) or Cheap Cloud.
Cost: ~$50 (for running benchmarks).

Day 15-17: The Benchmarks

Benchmark 1 (Accuracy): Run your model vs. Base R1-Distill-8B on the MATH Test Set.

Benchmark 2 (Robustness): Run the "Crash Test". How often does it produce invalid XML?

Benchmark 3 (Ablation): Disable the recursion (force linear generation) and measure the drop in accuracy. This is your strongest table in the paper.

Day 18-25: Writing the Paper

Focus: Results, Graphs (Accuracy vs Recursion Depth), and Abstract.

Visuals: Create a clear diagram of the <elaborate> -> Controller -> Resume loop.

Day 26-28: Submission

Format for arXiv or your target conference.

# Project

This paper follows from previous inspiration on recursive LLM (rLLM) where LLM can write a code to call another LLM. The following architecture does not need the LLM to learn to write python, it is recursive in nature.

Just like reasoning models puts their thoughts in the `<think> </think>` tag before they out put, here I introduce two new tags for this model `<elaborate> </elaborate>`, `<response> </response>`and `</run>`.

The AI works like the flowing:

```html
Input: write code for me on xxx
Output:
<think>
We break target into A, B C
<elaborate>
write code for A
</elaborate>
<elaborate>
write code for B
</elaborate>
<elaborate>
write code for C
</elaborate>
</run>
<elaborate>
debug code for ...
</elaborate>
</run>
</think>
Here is the code for xxx: ...
```

- It can plan its response in the thinking tag
- whenever it needs to dive into some problem, it can use `<elaborate>
</elaborate>` to call another AI
- whenever the `</run>` tag is outputed, the AI will send the text in the `<elaborate>
</elaborate>` tag to another AI in parallel and the response will be in the `<response> </response>` block. - Note that `<elaborate></elaborate>` calls the same AI as the root AI, so it can further recurse. The input is the text in the tags - The `<elaborate></elaborate>` block will be deleted after getting the response - The `</run>` block will also be deleted after getting the response
- Everything in the `<think></think>` tag is not visable to user or upper level AI, only the text after will be returned

So here is the run:

```html
<think>
We break target into A, B C
<elaborate>
write code for A
</elaborat>
<elaborate>
write code for B
</elaborate>
<elaborate>
write code for C
</elaborate>
</run>
```

```html
<think>
    We break target into A, B C
    <response> xxx </response>
    <response> yyy </response>
    <response> zzz </response></think
>
```

```html
<think>
We break target into A, B C
<response>
xxx
</response>
<response>
yyy
</response>
<response>
zzz
</response>
<elaborate>
debug code for xxx,yy,zzz
</elaborate>
</run>
```

```html
<think>
We break target into A, B C
<response>
xxx
</response>
<response>
yyy
</response>
<response>
zzz
</response>
<response>
no bug found
</response>
</run>
</think>
Here is the code for xxx: xxx,yyy,zzz
```

Advantage of this model:

- It don’t “run” recursion on python, so it follows a less strict grammar yet still have recursive ability
- Why need `</run>` tag? it need parallelism for speed and serial processing. If recursion happens simultaneously, it cannot benefit from previous response. If we recurse and replace one by one, it is a extra long chain of thought and it will be slow.
- This is “Turing Complete”, it can process a single sentece, it has condition (if previous response is xxx, we do yyy), it has loop (use if as gate keep elaborating until something is done).

It techiniquely can recover all major agent paradigms if it has tool call ability

- Reflection: it has this abiility in nature, see example above
- Tool use
- React = reflection + tool use
- planing: it also have this ability in the example above
- multi-agent: parallel calling different agents

## Training

- use GRPO to optimize the following
    - achieve high score in the test set
    - minimize thinking length (this encourage model to give thinking to lower level agents)
    - minimize tree depth and total thinking time
        - this encourage parallelism and a wider tree
        - prevents the model from passing all input to the next model, having 0 thinking making tree depth infinite
        - make sure the amount of thinking is balanced between each sub-model
- formating checking to make sure it is parsed correctly
- SFT
