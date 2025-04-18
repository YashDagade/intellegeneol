# RegenerateEOL: Running the Experiments

## Overview

RegenerateEOL is a novel approach to sentence embedding that generates diverse embeddings directly from the language model, without requiring complex sentence transformations. This repository contains the implementation of the RegenerateEOL approach described in the accompanying paper.

## What's New in RegenerateEOL

1. **Direct Diverse Embeddings**: Instead of generating multiple sentence transformations, RegenerateEOL directly generates 5 diverse embeddings for each input sentence.
2. **Diversity Signaling**: Each new embedding request includes information about previous embeddings to ensure true diversity.
3. **Token Efficiency**: RegenerateEOL uses ~500 tokens per sentence vs. ~10,000 for GenEOL's 32 transformations.
4. **Interactive Refinement**: The approach simulates a conversation where the model is asked to provide different perspectives.

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- MTEB (Massive Text Embedding Benchmark)
- Accelerate

## Setup

1. Clone the repository (if you haven't already):
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Experiments

We've simplified the process to run only the r5 method (5 diverse embeddings):

### Using the submit.sh script (recommended for slurm environments)

This option is recommended if you have access to a SLURM cluster:

```bash
chmod +x submit.sh
sbatch submit.sh
```

The `submit.sh` script will:
1. Submit SLURM jobs to run RegenerateEOL r5 on STS benchmarks
2. Use 5 diverse embeddings per sentence
3. Save results to the `Nlogs/RegenerateEOL_r5` directory

## What Happens During Execution

When you run the experiments, the following will happen:

1. For each sentence in the benchmark datasets:
   - The model generates an initial embedding using the "means in one word" technique
   - The model then generates 4 more diverse embeddings, each with knowledge of previous embeddings
   - All 5 embeddings are combined into a final representation

2. The console will show output like this:
```
Processing sentence 0/850: "The cat sat on the mat."
  Initial embedding for "The cat sat on the mat.": feline
  Diverse embedding #2: resting
  Diverse embedding #3: domestic
  Diverse embedding #4: position
  Diverse embedding #5: comfort
  Generated 5 diverse embeddings for sentence 0
```

3. Results will be saved to the appropriate output directories for analysis

## Output

The results will be saved in the following directory:

- `./resultsTF/regenerateol_r5_mistralai/Mistral-7B-Instruct-v0.1_1_1/mistralai/Mistral-7B-v0.1`

## Analyzing Results

After running the experiments, you can analyze the results using the provided analysis tools:

```bash
python gather_sts_results.py --create_viz
```

This will generate CSV files with the benchmark results and visualizations in the `sts_visualizations` directory.

## Method Parameters

The r5 implementation uses these key parameters:

- `--method r5`: Use RegenerateEOL with 5 diverse embeddings
- `--normalized`: Apply L2 normalization to embeddings
- `--seed 42`: Set random seed for reproducibility
- `--select 5`: Use all 5 generated embeddings

## Implementation Details

### Key Files Modified

1. **geneol/prompts_utils.py**:
   - Added `get_task_specific_gen_prompt` function
   - Enhanced `get_regenerate_emb_prompts` to include previous embedding context
   - Added conversation-style prompting patterns

2. **geneol/geneol.py**:
   - Added r5 method implementation
   - Implemented sequential embedding with diversity signaling
   - Enhanced logging to show embedding words
   - Optimized embedding aggregation

3. **submit.sh**:
   - Simplified to run only r5 method
   - Added clear logging and configuration

## Troubleshooting

If you encounter any issues running the experiments:

1. Check the SLURM output logs in the `Nlogs/RegenerateEOL_r5` directory
2. Make sure the model paths are correct
3. Try running with a smaller batch size if you encounter OOM errors
4. Ensure you have the latest version of all dependencies

## Expected Results

While full benchmark results are pending, we expect RegenerateEOL to:
1. Achieve comparable or better performance than ContrastiveEOL (~0.765 on STS benchmarks)
2. Use significantly fewer tokens (5 embeddings vs. 32 transformations)
3. Show greater robustness through embedding diversity 