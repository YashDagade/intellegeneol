# IntelleGenEOL: Intelligent Transformation for Generation-Enhanced One-word Latent Spaces

## Overview

IntelleGenEOL improves upon the GenEOL approach by introducing a more intelligent transformation pipeline. Instead of generating many random transformations of a sentence (e.g., 32 variations), IntelleGenEOL uses a three-stage process:

1. **Thinker**: Analyzes the input sentence and determines the optimal transformation strategy
2. **Generator**: Applies the chosen strategy to generate high-quality transformations
3. **Embedder**: Embeds the original and transformed sentences and combines them

This approach aims to produce better embeddings with fewer transformations, reducing computational cost while improving quality.

## Implementation Details

### Thinker Stage

The Thinker LLM analyzes each sentence and determines which transformation strategy would best capture its semantic meaning:

- **Elaboration**: Add more context or details that clarify the meaning
- **Simplification**: Reduce complexity while preserving core meaning
- **Specificity**: Replace general terms with more specific ones
- **Contextual framing**: Add domain-specific context
- **Abstraction**: Express the underlying concept at a higher level
- **Paraphrasing**: Restate using different words but same meaning

For task-specific scenarios, the Thinker also takes into account the nature of the task (e.g., sentiment classification, question answering).

### Generator Stage

The Generator LLM receives the original sentence along with the chosen strategy and creates 5 variations of the sentence applying that strategy. This targeted approach produces more meaningful transformations than random ones.

### Embedder Stage

The Embedder LLM (which can be the same as the Generator) embeds each sentence with a "means in one word:" prompt, and the embeddings are averaged to create the final representation.

## Key Changes

1. Added new prompt functions:
   - `get_thinker_prompt()`: Creates prompts for the Thinker LLM
   - `get_generator_prompt()`: Creates strategy-specific prompts for the Generator LLM

2. Added a new method `t5` in the GenEOL class:
   - Implements the three-stage process
   - Includes error handling and fallbacks
   - Uses JSON formatting for structured outputs

3. Updated submission scripts to use the new method:
   - Changed default method from `s5` to `t5`
   - Updated task names and output directories

## Expected Improvements

The IntelleGenEOL approach should provide several benefits:

1. **Higher quality embeddings**: By using optimal transformation strategies for each sentence, we generate more semantically relevant variations.

2. **Better task performance**: Task-specific context helps the Thinker choose strategies that are most appropriate for each application.

3. **Reduced computation**: Using 5 high-quality transformations instead of 32 random ones significantly reduces computational requirements.

4. **Improved robustness**: The approach should work better across different types of sentences, including complex, ambiguous, or domain-specific text.

## Usage

To run the IntelleGenEOL experiments:

```bash
sbatch submit.sh
```

This will execute the pipeline with the Mistral-7B model family on the specified benchmark tasks.

## Results

Results will be saved in the `./Nlogs/intelleGenEOL_t5_mistral_instr_8b_5_1_8/` directory, organized by embedding model, select value, and seed. 