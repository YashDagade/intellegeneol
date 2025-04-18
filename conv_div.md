# Conversational Diversity (conv_div) for GenEOL

## Concept

Conversational Diversity improves embeddings by exploring a wider semantic space through hierarchical generation and clustering. Instead of generating all variations at once, we use a two-step process with clustering in between to ensure diversity.

## Workflow

1. **Initial Generation**: Generate 16 variations of the input text
2. **Clustering**: Group these 16 variations into 4 clusters of 4 each
3. **Centroid Selection**: Identify the most representative text (centroid) from each cluster
4. **Secondary Generation**: Generate 4 new variations for each centroid (16 total)
5. **Final Embedding**: Create the final embedding by averaging these diverse variations

## Benefits

- More diverse semantic space exploration
- Better representation of different aspects of the input
- Prevents semantic collapse where variations are too similar
- Improved performance on semantic tasks

## Implementation Details

The implementation modifies the GenEOL encode method to support this hierarchical generation process, with careful logging of all generations for interpretability.

### Step-by-Step Process

1. **Initial Generation (16 variations)**
   - Start with the original input text
   - Generate 15 variations using the specified method (s5, d5, etc.)
   - Include the original text as one of the 16 variations

2. **Embedding for Clustering**
   - Embed all 16 variations using the embedding model
   - Use task-specific prompts if enabled

3. **Clustering**
   - Compute pairwise cosine similarities between all embeddings
   - Use Agglomerative Clustering with appropriate parameters based on scikit-learn version
   - Form 4 clusters with approximately equal sizes

4. **Centroid Selection**
   - For each cluster, compute the mean embedding
   - Select the variation closest to this mean as the centroid
   - This represents the most "central" or representative example in each cluster

5. **Secondary Generation**
   - For each of the 4 centroids, generate 3 new variations
   - Include the centroid itself as the 4th variation
   - This gives 16 final variations (4 variations × 4 centroids)

6. **Final Embedding**
   - Embed all 16 final variations
   - Average these embeddings to create the final representation

### Clustering Approach

We use Agglomerative Clustering to ensure balanced clusters (4 clusters with 4 samples each). The implementation is version-agnostic and will work with different scikit-learn versions:

```python
# First try newer scikit-learn version
try:
    clustering = AgglomerativeClustering(
        n_clusters=4,
        affinity='precomputed',
        linkage='average'
    ).fit(distance_matrix)
except TypeError:
    # Fall back to older scikit-learn version
    try:
        clustering = AgglomerativeClustering(
            n_clusters=4,
            linkage='average',
            connectivity=None,
            compute_full_tree='auto'
        ).fit(distance_matrix)
    except TypeError:
        # If that also fails, use even simpler version
        clustering = AgglomerativeClustering(
            n_clusters=4
        ).fit(embeddings_np)
```

Centroid selection is done by finding the sample closest to the mean embedding of each cluster:

```python
# Calculate the mean embedding for the cluster
mean_emb = np.mean(cluster_embeddings, axis=0)

# Find the closest example to the mean
distances = [np.linalg.norm(mean_emb - emb) for emb in cluster_embeddings]
centroid_idx = np.argmin(distances)
centroid = cluster_texts[centroid_idx]
```

## Logging and Analysis

All generations are logged in detail for analysis:

1. **Initial Variations**: All 16 variations generated in the first step
2. **Clusters**: Which variations belong to which clusters
3. **Centroids**: The 4 selected centroids (one from each cluster)
4. **Final Variations**: All 16 variations used for the final embedding

This information is saved in JSON format at:
`{output_folder}/../transformations/{task}_conv_div_generations{encode_call}_{process_index}.json`

## Usage

To use Conversational Diversity, add the `--conv_div` flag to your GenEOL command:

```bash
sbatch submit.sh  # Submit script already includes the --conv_div flag
```

## Expected Improvements

The conv_div approach should improve performance on:

1. **Semantic textual similarity tasks**: By better capturing diverse aspects of meaning
2. **Retrieval tasks**: Through more robust representations that explore the semantic space
3. **Classification tasks**: By including diverse perspectives in the embedding

The improvement should be most noticeable when the input text has multiple interpretations or dimensions that need to be represented in the embedding.
