#!/bin/bash

# Configuration for RegenerateEOL experiments
select_value=5  # Number of diverse embeddings to generate

# Random seed for reproducibility
seed=42

# Models and their corresponding output subdirectories
declare -A models
models["mistralai/Mistral-7B-v0.1"]="mistral0.1" # this is the embedding model

# Base directories and other parameters
base_output_dir="./Nlogs/RegenerateEOL_r5"
base_script="./scripts/PromptEMB_accelerate_mteb.sh"
partition="compsci-gpu"
array="0-8%10"  # Array job for STS tasks
gres="gpu:a5000:1"
ntasks=1
mem="40gb"
gen_model="mistralai/Mistral-7B-Instruct-v0.1"
session="r5"  # Using r5 method (5 diverse embeddings)
gpu_count=1
task_per_node=1

# Print banner
echo "===================================================="
echo "Launching RegenerateEOL (r5) experiments"
echo "===================================================="
echo "Using 5 diverse embeddings per sentence"
echo "Embedding model: ${models[@]}"
echo "Generation model: $gen_model"
echo "Output directory: $base_output_dir"
echo "===================================================="

# Iterate over the models
for emb_model in "${!models[@]}"; do
  model_subdir="${models[$emb_model]}"

  # Set up output directory
  output_dir="${base_output_dir}/${model_subdir}_k${select_value}_seed${seed}"
  
  # Create the output directory if it doesn't exist
  mkdir -p "$output_dir"

  echo "Submitting RegenerateEOL job with method r5, embedding model $emb_model, select value $select_value, seed $seed"
  sbatch --partition=$partition --array=$array --gres=$gres --ntasks=$ntasks --mem=$mem \
         --output="${output_dir}/%03a.out" \
         $base_script regenerateol $session $gen_model $emb_model $gpu_count $task_per_node \
         --normalized --select $select_value --seed $seed
done

echo "===================================================="
echo "All jobs submitted! Check the logs in $base_output_dir"
echo "====================================================" 