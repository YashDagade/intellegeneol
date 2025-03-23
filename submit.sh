#!/bin/bash

# Array of select values
select_values=(24)
# select_values=(2)

# Array of seed values
seed_values=(42)
# seed_values=(42)

# Models and their corresponding output subdirectories
declare -A models
models["mistralai/Mistral-7B-v0.1"]="mistral0.1" # this is the embedding model
# models["meta-llama/Meta-Llama-3-8B"]="llama3"

# Base directories and other parameters
base_output_dir="./Nlogs_meta/1014AB2_comp2_s5_llama3.1_8b_1_1_8"
base_script="./scripts/PromptEMB_accelerate_mteb.sh"
partition="compsci-gpu"
array="0-8%10"
gres="gpu:a5000:1"
ntasks=1
mem="40gb"
model_1="mistralai/Mistral-7B-Instruct-v0.1"
task_name="llama_ethan_run"
session="s5"
gpu_count=1
task_per_node=2 # this is m - we need to change this to be optimal

# Outer loop: iterate over the models
for model_2 in "${!models[@]}"; do
  model_subdir="${models[$model_2]}"

  # Middle loop: iterate over the select values
  for select_value in "${select_values[@]}"; do

    # Inner loop: iterate over the seed values
    for seed in "${seed_values[@]}"; do
      # Include the seed value in the output directory
      output_dir="${base_output_dir}/${model_subdir}_k${select_value}_seed${seed}"

      sbatch --partition=$partition --array=$array --gres=$gres --ntasks=$ntasks --mem=$mem \
             --output="${output_dir}/%03a.out" \
             $base_script $task_name $session $model_1 $model_2 $gpu_count $task_per_node \
             --compositional --select $select_value --seed $seed
    done
  done
done