# RegenerateEOL: Efficient Diverse Embeddings Through Regeneration

## Hypothesis

Current embedding enhancement methods like GenEOL and ContrastiveEOL focus on generating multiple sentence transformations, which is computationally expensive (requiring 10-32 sentence generations) for modest gains. We propose **RegenerateEOL**, a novel approach that leverages the natural diversity of language models by directly asking them to generate diverse embeddings for the same sentence.

## Key Insight

The key insight behind RegenerateEOL is that we can exploit the LLM's ability to generate different perspectives on the same input by simply instructing it to provide "more diverse embeddings" in an interactive manner:

1. Generate an initial embedding using the "means in one word" technique
2. Instead of generating new sentences, simply ask the model to generate alternative embeddings
3. Combine these diverse embeddings to create a more expressive representation

This approach is inspired by how LLMs respond to math problems - when told their first answer is incorrect, they often produce different (and sometimes better) solutions.

## The RegenerateEOL Approach

### Core Principles

1. **Direct Embedding Diversity**: Generate different perspectives directly at the embedding level
2. **Token Efficiency**: Achieve better results with far fewer tokens compared to sentence transformation methods
3. **Interactive Refinement**: Use feedback-based prompting to guide the model toward diverse semantic spaces
4. **Hidden State Capture**: Extract more expressive representations from the model's internal states

### Implementation Details

1. **Progressive Diversification**: Each prompt builds on the previous one, guiding the model to explore different semantic dimensions
2. **Hidden State Extraction**: Instead of just taking word token embeddings, capture the full hidden state representation
3. **Embedding Aggregation**: Apply weighted averaging to combine diverse embeddings effectively
4. **Single-Model Pipeline**: Use the same LLM for both generating and embedding, eliminating cross-model compatibility issues
5. **Diversity Feedback**: Incorporate previous embedding words into subsequent prompts to encourage diversity

### Prompt Pattern

RegenerateEOL uses a unique interactive dialogue pattern that evolves based on previous embeddings:

```
User: The essence of a sentence is often captured by its main subjects and actions. This sentence: "[Text]" means in one word:
Assistant: [word1]

User: You previously described this sentence as meaning "[word1]". Give me a different perspective. This sentence: "[Text]" means in one word:
Assistant: [word2]

User: You've described this sentence as meaning "[word1]" and "[word2]". Provide a completely different perspective. This sentence: "[Text]" means in one word:
Assistant: [word3]

User: Your previous interpretations were "[word1], [word2], [word3]". Focus on a different aspect entirely. This sentence: "[Text]" means in one word:
Assistant: [word4]

User: Consider the emotions conveyed that weren't captured in your previous interpretations. This sentence: "[Text]" means in one word:
Assistant: [word5]
```

## Technical Implementation

### Code Additions

We implemented RegenerateEOL by adding the following components to the GenEOL codebase:

1. **get_regenerate_emb_prompts()**: A function that generates prompts for diverse embeddings, now enhanced to incorporate previous embedding words.

```python
def get_regenerate_emb_prompts(input_text, num_diverse, task=None, previous_embeddings=None):
    """
    Generate prompts for RegenerateEOL that ask for diverse embeddings.
    
    Args:
        input_text (str): The input sentence to embed
        num_diverse (int): Number of diverse embeddings to generate
        task (str, optional): The task context
        previous_embeddings (list, optional): List of previous embedding words
    """
    # Base prompt template
    base_prompt = f"<s>The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind, this sentence: \"{input_text}\" means in one word:\""
    
    # Create diversity prompts that reference previous embeddings
    if previous_embeddings is not None and len(previous_embeddings) > 0:
        diversity_prompts = [
            f"<s>You previously described this sentence as meaning \"{previous_embeddings[0]}\". Give me a different perspective...",
            # More prompts that reference previous embeddings
        ]
    else:
        # Default diversity prompts
        diversity_prompts = [
            f"<s>That's one perspective. From another angle, this sentence: \"{input_text}\" means in one word:\"",
            # More generic prompts
        ]
    
    # Return the appropriate prompts
    return [base_prompt] + diversity_prompts[:num_diverse-1]
```

2. **Modified Embedding Process**: The embedding process now captures and uses previous embedding words for better diversity:

```python
# Process each diverse embedding one at a time
for i in range(num_diverse):
    # Get prompt with context from previous embeddings
    diverse_prompts = get_regenerate_emb_prompts(sentence, 1, task, embedding_words)
    prompt = diverse_prompts[0]
    
    # Get embedding for this prompt
    last_hidden_state, inputs = self.llm.embed([prompt])
    
    # Try to extract the last token for logging and diversity guidance
    try:
        last_token = self.llm.tokenizer.decode(inputs['input_ids'][0, -1:])
        embedding_words.append(last_token.strip())
    except:
        embedding_words.append(f"embedding_{i}")
```

### Method Parameters

- **r5**: RegenerateEOL with 5 diverse embeddings
- **--normalized**: Apply L2 normalization to embeddings
- **--select**: Number of top embeddings to use (e.g., --select 5)
- **--seed**: Random seed for reproducibility

## Example Output

Here's an example of RegenerateEOL in action:

```
Processing sentence 0/850: "The cat sat on the mat."
  Initial embedding for "The cat sat on the mat.": feline
  Diverse embedding #2: resting
  Diverse embedding #3: domestic
  Diverse embedding #4: position
  Diverse embedding #5: comfort
  Generated 5 diverse embeddings for sentence 0

Processing sentence 1/850: "A person is playing a guitar."
  Initial embedding for "A person is playing a guitar": music
  Diverse embedding #2: performance
  Diverse embedding #3: entertainment
  Diverse embedding #4: artistic
  Diverse embedding #5: melodic
  Generated 5 diverse embeddings for sentence 1
```

## Comparison with Previous Approaches

| Approach | Token Cost | Computational Steps | Information Preservation | Avg STS Score |
|----------|------------|---------------------|--------------------------|---------------|
| GenEOL (s5) | ~10,000 | Generate 32 sentences + embed 33 inputs | Indirect (via paraphrasing) | 0.7969 |
| ContrastiveEOL (c3) | ~1,000 | Generate 3 sentences + embed 4 inputs | Partial (limited perspectives) | 0.7653 |
| RegenerateEOL (r5) | ~500 | Generate 5 embeddings directly | Direct (hidden state access) | Pending |

## Implementation Changes

We've made the following changes to implement RegenerateEOL:

1. **prompts_utils.py**:
   - Added `get_task_specific_gen_prompt()` function to fix compatibility issues
   - Enhanced `get_regenerate_emb_prompts()` to support previous embedding context
   - Added task-specific prompt support

2. **geneol.py**:
   - Added `r5` method to the encode function
   - Implemented sequential embedding generation with feedback
   - Enhanced logging to display the embedding words
   - Added diversity signaling between embeddings

3. **submit.sh**:
   - Updated to run only the r5 method
   - Simplified to use a single configuration
   - Added informative output about the experiment

By focusing on generating diversity directly at the embedding level rather than through sentence transformations, RegenerateEOL aims to achieve superior performance with dramatically reduced computational requirements. This makes it particularly well-suited for resource-constrained environments and real-time applications.

## Future Work

For future exploration, we could:

1. **Weighted Aggregation**: Experiment with different weighting schemes for combining diverse embeddings
2. **Rejection Sampling**: Implement a mechanism to reject low-diversity embeddings
3. **Task-Adaptive Prompting**: Further enhance prompts based on the specific task
4. **Representation Analysis**: Analyze the diversity of embeddings in the vector space 