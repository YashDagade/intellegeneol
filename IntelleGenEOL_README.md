# IntelleGenEOL: Embedding-Optimized Language Generation

## Overview

IntelleGenEOL is an improved approach to generate high-quality sentence transformations specifically optimized for embedding models. The approach uses a two-stage reasoning process to create semantically rich transformations that enhance embedding quality.

## Two-Stage Process

### 1. Reasoning Stage

First, the system analyzes the input sentence to determine the optimal transformation strategy. It considers:

- The key semantic concepts and relationships
- Any ambiguities, vagueness, or missing information
- The complexity level and density of information
- The context or domain the sentence belongs to

Based on this analysis, it selects one of six transformation strategies:

- **Elaboration**: Adding explanatory details that clarify concepts
- **Simplification**: Reducing complexity while preserving core meaning
- **Specificity**: Replacing vague terms with more precise language
- **Contextual framing**: Adding domain context to disambiguate meaning
- **Abstraction**: Extracting higher-level concepts from specific details
- **Paraphrasing**: Restating using different but semantically equivalent phrasing

### 2. Transformation Stage

Once the optimal strategy is determined, the system applies it to create a transformation that:

- Preserves or enhances the semantic meaning of the original sentence
- Addresses weaknesses in the original sentence
- Improves representation in the embedding space

Each transformation strategy is accompanied by carefully designed examples to guide the model.

## Implementation

The implementation uses two key functions:

1. `get_thinker_reasoning_prompt()`: Generates a prompt for analyzing the input sentence and determining the optimal transformation strategy
2. `get_transformation_debug_prompt()`: Generates a prompt for transforming the sentence based on the chosen strategy and reasoning

The system provides detailed debug information during execution:
- Original sentence
- Selected strategy
- Reasoning behind the strategy
- Transformed sentence

## Using IntelleGenEOL

To run the IntelleGenEOL pipeline:

```bash
sbatch submit.sh
```

This will execute the pipeline with default settings:
- Task: 1 (STS)
- Model: Mistral-7B-Instruct-v0.1
- Method: t1 (1 transformation per sentence)
- Select: 2 (using top 2 sentences for evaluation)

## Customization

You can customize the pipeline by editing the `submit.sh` file:

- Change the `task` variable to target different MTEB tasks
- Modify the `model_name` to use different LLMs
- Adjust `select` to use a different number of sentences for evaluation
- Change `base_output_dir` to save results in a different location

## Advantages Over Previous Approaches

IntelleGenEOL offers several advantages:

1. **Strategic Transformations**: Uses task-specific reasoning to select the optimal transformation strategy
2. **Explicit Reasoning**: Provides transparent reasoning behind transformation choices
3. **Debugging Support**: Includes detailed debugging information to track transformation quality
4. **Robustness**: Avoids complex JSON parsing that might fail with LLM outputs
5. **Efficiency**: Generates one high-quality transformation rather than multiple potentially redundant ones 