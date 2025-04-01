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

## Experimental Results

Based on our experimental evaluation across STS benchmark tasks:

```
config_id                                                      STS12     STS13     STS14     STS15     STS16     STS17     STSBenchmark  SICK-R    AVG_STS     
1014AB2_comp2_s5_Mistral-7B-Instruct-v0.1_Mistral-7B-v0.1_kx_sx,0.696361,0.83619,0.776004,0.822641,0.814704,0.821881,0.810465,0.765485,0.7968922857
contrastiveeol_c3_Mistral-7B-Instruct-v0.1_Mistral-7B-v0.1_kx_sx,0.628089,0.809002,0.739779,0.809173,0.782004,0.811719,0.777597,0.738142,0.7653375714
baseline_b5_Mistral-7B-Instruct-v0.1_Mistral-7B-v0.1_kx_sx,0.602343,0.78688,0.695964,0.768002,0.765251,0.812551,0.751918,0.735104,0.7404155714
intellegeneol_t1_Mistral-7B-Instruct-v0.1_Mistral-7B-v0.1_kx_sx,0.575225,0.78097,0.709215,0.788521,0.764869,0.81124,0.744178,,0.739174
```

The ContrastiveEOL approach (c3) achieved an average STS score of 0.7653, which outperforms both the baseline EOL approach (b5) with 0.7404 and the IntelleGenEOL approach (t1) with 0.7391. However, it still falls short of the original GenEOL approach (s5) which scored 0.7969.

### Observations from Generated Transformations

Examples of transformations show that the approach generally achieves its goal of creating concise, semantically rich variations:

1. Original: "A cat is eating some corn."  
   Semantic Core: "Corn is being consumed by a feline."  
   Perspective Shift: "Some corn is being eaten by a cat."  
   Implicated Information: "A cat is currently consuming corn as its meal."

2. Original: "A man is playing a guitar."  
   Semantic Core: "A person is strumming a stringed instrument."  
   Perspective Shift: "A guitar is being played by a person."  
   Implicated Information: "A man is currently engaging in the activity of playing a guitar."

3. Original: "Darren dopp, a spitzer spokesman, declined to comment late thursday."  
   Semantic Core: "Darren Dopp refused to comment on Thursday."  
   Perspective Shift: "On Thursday, Darren Dopp, a spokesman for Spitzer, declined to comment."  
   Implicated Information: "Spitzer spokesperson Darren Dopp remained silent on the matter on Thursday."

## Limitations and Areas for Improvement

While ContrastiveEOL showed promise, several limitations were identified:

1. **Limited Semantic Diversity**: The three transformation types sometimes produce variations that are too similar to each other, limiting the benefit of multiple transformations.

2. **Occasional Errors**: The model sometimes introduces factual errors or misinterpretations in transformations (e.g., "incident was caused by the sonic boom of the Concorde's son").

3. **Insufficient Semantic Enrichment**: While the approach is more efficient than GenEOL's 32 transformations, it may not provide enough semantic dimensionality to match GenEOL's performance.

4. **Inconsistent Transformation Quality**: The quality of transformations varies across different input types, with complex sentences sometimes receiving more benefit than simple ones.

5. **Perspective Shift Limitations**: The passive voice transformations in the Perspective Shift category often contribute minimal semantic diversity.

## Conclusion and Next Steps

The ContrastiveEOL approach demonstrates that strategic, minimal transformations can improve embedding quality over baseline methods while being more computationally efficient than the original GenEOL approach. However, further refinement is needed to match or exceed GenEOL's performance.

Future directions should focus on:
1. Increasing semantic diversity while maintaining conciseness
2. Implementing more sophisticated transformation selection mechanisms
3. Exploring hybrid approaches that combine the efficiency of ContrastiveEOL with the performance of GenEOL 