#!/bin/bash

# Array of select values
select_values=(5)

# Array of seed values
seed_values=(42)

# Models and their corresponding output subdirectories
declare -A models
models["mistralai/Mistral-7B-v0.1"]="mistral0.1" # this is the embedding model
# models["meta-llama/Meta-Llama-3-8B"]="llama3"

# Base directories and other parameters
base_output_dir="./Nlogs/DiverseGenEOL_d5_mistral_instr_5_1_1"
base_script="./scripts/PromptEMB_accelerate_mteb.sh"
partition="compsci-gpu"
array="0-8%10"
gres="gpu:a5000:1"
ntasks=1
mem="40gb"
thinker_model="mistralai/Mistral-7B-Instruct-v0.1"
gen_model="mistralai/Mistral-7B-Instruct-v0.1"
task_name="diversegeneol"
session="d5"
gpu_count=1
task_per_node=1 # this is m - we need to change this to be optimal

# Outer loop: iterate over the models
for emb_model in "${!models[@]}"; do
  model_subdir="${models[$emb_model]}"

  # Middle loop: iterate over the select values
  for select_value in "${select_values[@]}"; do

    # Inner loop: iterate over the seed values
    for seed in "${seed_values[@]}"; do
      # Include the seed value in the output directory
      output_dir="${base_output_dir}/${model_subdir}_k${select_value}_seed${seed}"
      
      # Create the output directory if it doesn't exist
      mkdir -p "$output_dir"

      echo "Submitting DiverseGenEOL job with method d5, embedding model $emb_model, select value $select_value, seed $seed"
      sbatch --partition=$partition --array=$array --gres=$gres --ntasks=$ntasks --mem=$mem \
             --output="${output_dir}/%03a.out" \
             $base_script $task_name $session $gen_model $emb_model $gpu_count $task_per_node \
             --compositional --select $select_value --seed $seed
    done
  done
done 