# ContrastiveEOL: Embedding-Optimized Language with Semantic Contrasts

## Hypothesis

The current approaches to sentence embedding enhancement (GenEOL, IntelleGenEOL) have shown mixed results, with some improvements but also significant limitations. Based on our observations, we propose a new approach called **ContrastiveEOL** that focuses on generating semantically distinct but conceptually related transformations that create meaningful contrasts in the embedding space.

## Key Insights from Previous Approaches

From our experiments with IntelleGenEOL, we observed several limitations:

1. Verbose, ornate transformations appear to dilute rather than enhance semantic meaning
2. Complex multi-stage reasoning has high computational cost but modest benefits
3. Deliberate strategy selection doesn't consistently outperform simpler approaches
4. The "means in one word" prompt may already capture key semantics

## The ContrastiveEOL Approach

ContrastiveEOL builds on these insights to create a more efficient approach:

### Core Principles

1. **Semantic Triangulation**: Generate transformations that approach the same meaning from different semantic angles
2. **Minimalism**: Create concise transformations that preserve core meaning without verbosity
3. **Controlled Diversity**: Use carefully designed prompts that provide the right balance of similarity and difference
4. **Single-pass Generation**: Generate all transformations in a single LLM call

### Transformation Types

ContrastiveEOL uses three specific transformation types:

1. **Semantic Core**: Extract the most essential meaning (e.g., "Cats eating" → "Felines consuming food")
2. **Perspective Shift**: Present the same information from a different viewpoint (e.g., "The food is being eaten by kittens")
3. **Implicated Information**: Include information that is implied but not stated (e.g., "Young cats satisfying their hunger")

### Implementation Details

1. **Efficient Prompting**: A single prompt will instruct the LLM to generate all three transformation types at once
2. **Balanced Embedding**: Equal weight is given to original and transformed sentences
3. **Vector Operations**: Use averaging and other vector operations to combine embeddings effectively
4. **Single-Stage Process**: Eliminate the computationally expensive reasoning stage

## Expected Benefits

1. **Computational Efficiency**: 3-4x fewer LLM calls than GenEOL and significantly fewer than IntelleGenEOL
2. **Semantic Richness**: A more complete representation of meaning through triangulation
3. **Reduced Noise**: Avoiding verbose transformations reduces semantic drift
4. **Better Transferability**: By focusing on core meaning rather than surface details, embeddings should transfer better across tasks

## Preliminary Implementation

The ContrastiveEOL approach will be implemented with the following components:

1. **Prompt Function**: `get_contrastive_transformations_prompt()`
2. **Processing Function**: `process_contrastive_outputs()`
3. **Method Parameter**: `c3` (for ContrastiveEOL with 3 transformation types)

## Evaluation Plan

To test the effectiveness of ContrastiveEOL, we will evaluate it against existing approaches:

1. **Baseline Performance**: Compare against vanilla LLM embeddings (method `b5`)
2. **GenEOL Comparison**: Compare against the original GenEOL approach (method `s5`)
3. **IntelleGenEOL Comparison**: Compare against the IntelleGenEOL approach (method `t1`)

Particular attention will be paid to STS tasks, where semantic nuance is most important. 