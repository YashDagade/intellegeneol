# Intelligent Transformation Approaches for Enhanced Sentence Embeddings

## Overview

This document describes approaches for enhancing sentence embeddings through intelligent transformation techniques. The core idea is to transform input sentences in ways that preserve or enhance their semantic meaning, resulting in better embeddings for downstream tasks.

## Transformation Approaches

### IntelleGenEOL

IntelleGenEOL uses a three-stage process to transform sentences:

1. **Thinker**: Analyzes the input sentence and determines the optimal transformation strategy
2. **Generator**: Applies the chosen strategy to generate high-quality transformations
3. **Embedder**: Embeds the original and transformed sentences and combines them

This approach aims to produce better embeddings with fewer transformations compared to approaches that generate many random variations.

### Two-Stage Process 

The two-stage reasoning process focuses on:

1. **Reasoning Stage**: Analyzes the input sentence to determine the optimal transformation strategy by considering:
   - Key semantic concepts and relationships
   - Ambiguities, vagueness, or missing information
   - Complexity level and information density
   - Context or domain

2. **Transformation Stage**: Applies the chosen strategy to create a transformation that:
   - Preserves or enhances semantic meaning
   - Addresses weaknesses in the original sentence
   - Improves representation in the embedding space

## Transformation Strategies

Six transformation strategies are available:

- **Elaboration**: Adding explanatory details that clarify concepts
- **Simplification**: Reducing complexity while preserving core meaning
- **Specificity**: Replacing vague terms with more precise language
- **Contextual framing**: Adding domain context to disambiguate meaning
- **Abstraction**: Extracting higher-level concepts from specific details
- **Paraphrasing**: Restating using different but semantically equivalent phrasing

## Implementation

The implementation uses several key components:

1. Prompt generation functions:
   - `get_thinker_reasoning_prompt()`: Creates prompts for analyzing the input sentence
   - `get_transformation_debug_prompt()`: Creates prompts for transforming sentences

2. Task-specific contexts that tailor transformations to different NLP tasks:
   - STS (Semantic Textual Similarity)
   - Retrieval
   - Classification
   - Reranking
   - Clustering
   - PairClassification
   - Summarization

3. Embedding methods:
   - Using "means in one word:" prompts
   - Averaging embeddings across transformations

## Experimental Findings

### Performance Analysis

Based on the experimental results:

```
config_id                                                        STS12     STS13     STS14     STS15     STS16     STS17     STSBenchmark  SICK-R    AVG_STS     
1014AB2_comp2_s5_Mistral-7B-Instruct-v0.1_Mistral-7B-v0.1_kx_sx  0.696361  0.83619   0.776004  0.822641  0.814704  0.821881  0.810465      0.765485  0.7968922857  
llama_ethan_run_s5_Llama-3.1-8B-Instruct_Mistral-7B-v0.1_kx_sx   0.655104  0.825289  0.767385  0.813974  0.803927  0.83658   0.788608      0.768061  0.7844095714  
intellegeneol_t1_Mistral-7B-Instruct-v0.1_Mistral-7B-v0.1_kx_sx  0.575225  0.78097   0.788521  0.764869  0.81124   0.744178  N/A           N/A       0.7441671667  
```

The intelligent transformation approach (`intellegeneol_t1`) underperforms compared to other methods like the standard GenEOL approach (`s5`).

### Observations from Generated Transformations

Examples of transformations show the model tends to produce verbose, overly complex sentences:

1. Original: "Some kittens are eating"  
   Strategy: Paraphrasing  
   Transformed: "Some feline offspring are partaking in the consumption of nourishment."

2. Original: "A person is singing and playing a guitar"  
   Strategy: Abstraction  
   Transformed: "A person is performing a multitask involving singing and playing a guitar."

3. Original: "A few animals are playing in the water"  
   Strategy: Contextual framing  
   Transformed: "In the context of a marine wildlife sanctuary, a few mammals are engaging in playful behavior in the crystal-clear waters."

### Key Limitations Identified

1. **Verbosity**: Transformations often add excessive detail that dilutes core meaning
2. **Semantic drift**: Some transformations stray too far from the original meaning
3. **Strategy selection**: The reasoning model may not consistently select optimal strategies
4. **Computational inefficiency**: The multi-stage pipeline requires multiple LLM calls
5. **Over-elaboration**: Transformations often add irrelevant details not present in the original

## Usage

To run the transformation pipeline:

```bash
sbatch submit.sh
```

This will execute the pipeline with the default settings:
- Model: Mistral-7B-Instruct-v0.1
- Method: t1 (1 transformation per sentence)
- Select: 2 (using top 2 sentences for evaluation)

You can customize the pipeline by editing the `submit.sh` file. 