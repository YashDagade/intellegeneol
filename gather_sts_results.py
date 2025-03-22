import os
import json
import re
import pandas as pd
import glob
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory
BASE_DIR = "/usr/project/xtmp/yd211/Documents/IntelleGENeol"

# Datasets to extract
STS_DATASETS = ["STS12", "STS13", "STS14", "STS15", "STS16", "STS17", "STS22", "STSBenchmark", "SICK-R"]

# Metrics to extract
METRICS = ["pearson", "spearman", "cosine_pearson", "cosine_spearman", 
           "manhattan_pearson", "manhattan_spearman", 
           "euclidean_pearson", "euclidean_spearman"]

def extract_config_from_path(path):
    """Extract configuration information from the result path."""
    # Extract the configuration part from the path
    # Example: resultsTF/llama_ethan_run_s5_Llama-3.1-8B-Instruct_1_2/Mistral-7B-v0.1
    
    # Get path relative to BASE_DIR
    rel_path = os.path.relpath(path, BASE_DIR)
    parts = rel_path.split(os.path.sep)
    
    if len(parts) < 3:
        return {
            "run_id": "unknown",
            "config": "unknown",
            "embedding_model": "unknown"
        }
    
    run_id = parts[1]  # e.g., llama_ethan_run_s5_Llama-3.1-8B-Instruct_1_2
    embedding_model = parts[2]  # e.g., Mistral-7B-v0.1
    
    # Try to extract more detailed config
    config_dict = {
        "run_id": run_id,
        "embedding_model": embedding_model
    }
    
    # Try to extract method, gen_model, batch_size, num_gens
    # Handles patterns like: llama_ethan_run_s5_Llama-3.1-8B-Instruct_1_2
    # And also: 1014AB2_comp2_s5_mistralchat0.1_1_1_8
    pattern1 = r"(.+)_(.+)_(.+)_(\d+)_(\d+)"
    pattern2 = r"(.+)_(.+)_(.+)_(.+)_(\d+)_(\d+)_(\d+)"
    
    match = re.match(pattern2, run_id)
    if match:
        config_dict["experiment"] = match.group(1)
        config_dict["comp_value"] = match.group(2)
        config_dict["method"] = match.group(3)
        config_dict["gen_model"] = match.group(4)
        config_dict["batch_size"] = match.group(5)
        config_dict["num_gens"] = match.group(6)
        config_dict["extra_param"] = match.group(7)
    else:
        match = re.match(pattern1, run_id)
        if match:
            config_dict["experiment"] = match.group(1)
            config_dict["method"] = match.group(2)
            config_dict["gen_model"] = match.group(3)
            config_dict["batch_size"] = match.group(4)
            config_dict["num_gens"] = match.group(5)
    
    # If there's a k and seed in the path, extract them too
    # Look for directories like mistral0.1_k24_seed42
    for part in parts:
        k_seed_pattern = r"(.+)_k(\d+)_seed(\d+)"
        match = re.match(k_seed_pattern, part)
        if match:
            config_dict["model_shortname"] = match.group(1)
            config_dict["k_value"] = match.group(2)
            config_dict["seed"] = match.group(3)
    
    return config_dict

def find_all_result_files():
    """Find all STS result files."""
    result_files = []
    
    # Pattern to match result directories
    patterns = [
        os.path.join(BASE_DIR, "resultsTF", "*", "*", "no_model_name_available", "no_revision_available", "*.json"),
        os.path.join(BASE_DIR, "resultsTF", "*", "*", "*.json")  # Some files might be directly in the model dir
    ]
    
    for pattern in patterns:
        result_files.extend(glob.glob(pattern))
    
    # Print some info about what we found
    print(f"Found {len(result_files)} JSON result files in total")
    
    return result_files

def extract_metrics(json_file):
    """Extract metrics from a JSON result file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract dataset name from the file
        dataset = os.path.basename(json_file).replace('.json', '')
        
        if 'scores' not in data:
            print(f"Warning: No 'scores' key in {json_file}")
            return None, None
            
        if 'test' not in data['scores']:
            # Check for 'validation' split instead (for STSBenchmark in dev mode)
            if 'validation' in data['scores']:
                scores = data['scores']['validation'][0]
            else:
                print(f"Warning: No 'test' or 'validation' split in {json_file}")
                return None, None
        else:
            scores = data['scores']['test'][0]
        
        # Extract metrics
        metrics_dict = {metric: scores.get(metric, float('nan')) for metric in METRICS}
        
        # Include the main_score if available
        if 'main_score' in scores:
            metrics_dict['main_score'] = scores['main_score']
        
        return dataset, metrics_dict
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None, None

def gather_all_results():
    """Gather all results and organize them into a dataframe."""
    result_files = find_all_result_files()
    all_results = []
    
    processed_count = 0
    for file_path in result_files:
        # Check if this is one of our target datasets
        dataset_name = os.path.basename(file_path).replace('.json', '')
        if dataset_name not in STS_DATASETS:
            continue
        
        # Extract configuration from path
        config = extract_config_from_path(file_path)
        
        # Extract metrics from file
        dataset, metrics = extract_metrics(file_path)
        if dataset is None or metrics is None:
            continue
        
        # Add dataset and metrics to configuration
        result_entry = config.copy()
        result_entry["dataset"] = dataset
        result_entry.update(metrics)
        
        all_results.append(result_entry)
        processed_count += 1
    
    print(f"Successfully processed {processed_count} STS benchmark files")
    return all_results

def create_pivot_tables(df):
    """Create pivot tables for different metrics."""
    pivot_tables = {}
    
    # Group metrics
    metric_groups = {
        "pearson": ["pearson", "cosine_pearson", "manhattan_pearson", "euclidean_pearson"],
        "spearman": ["spearman", "cosine_spearman", "manhattan_spearman", "euclidean_spearman"]
    }
    
    # Create a unique identifier for each run configuration
    # Include all the important configuration parameters
    df['config_id'] = df.apply(
        lambda row: f"{row.get('experiment', '')}_"
                   f"{row.get('method', '')}_"
                   f"{row.get('gen_model', '')}_"
                   f"{row.get('embedding_model', '')}_"
                   f"k{row.get('k_value', 'x')}_"
                   f"s{row.get('seed', 'x')}", 
        axis=1
    )
    
    # Create pivot tables for each metric group
    for group_name, metrics in metric_groups.items():
        for metric in metrics:
            # Create a pivot table with datasets as columns and configurations as rows
            pivot = df.pivot_table(
                index='config_id',
                columns='dataset',
                values=metric,
                aggfunc='first'  # Use first since we should only have one value per config_id/dataset
            )
            
            # Sort columns in a sensible order
            sorted_columns = sorted(
                pivot.columns, 
                key=lambda x: (
                    0 if x.startswith('STS') else 1,  # STS datasets first
                    0 if x == 'STSBenchmark' else 1,  # STSBenchmark before numbered ones
                    int(x[3:]) if x.startswith('STS') and x[3:].isdigit() else float('inf')  # Sort by number
                )
            )
            pivot = pivot[sorted_columns]
            
            # Add average across datasets
            sts_columns = [col for col in pivot.columns if col.startswith('STS')]
            pivot['AVG_STS'] = pivot[sts_columns].mean(axis=1)
            
            # Add average excluding certain datasets if needed
            stsb_columns = [col for col in sts_columns if col != 'STSBenchmark']
            if len(stsb_columns) > 0:
                pivot['AVG_STS_NO_BENCHMARK'] = pivot[stsb_columns].mean(axis=1)
            
            # Add this to our collection of pivot tables
            pivot_tables[f"{group_name}_{metric}"] = pivot
            
    return pivot_tables

def create_combined_table(df):
    """Create a single combined table with the most important metrics."""
    # Create a unique identifier for each run configuration
    df['config_id'] = df.apply(
        lambda row: f"{row.get('experiment', '')}_"
                   f"{row.get('method', '')}_"
                   f"{row.get('gen_model', '')}_"
                   f"{row.get('embedding_model', '')}_"
                   f"k{row.get('k_value', 'x')}_"
                   f"s{row.get('seed', 'x')}", 
        axis=1
    )
    
    # We'll use spearman as our main metric
    main_metric = 'spearman'
    
    # Create pivots for each dataset
    pivots = {}
    for dataset in STS_DATASETS:
        # Filter to just this dataset
        dataset_df = df[df['dataset'] == dataset]
        
        if dataset_df.empty:
            continue
            
        # Create pivot with just this dataset
        pivot = dataset_df.pivot_table(
            index='config_id',
            values=main_metric,
            aggfunc='first'
        )
        
        # Rename the column to the dataset name
        pivot.columns = [dataset]
        pivots[dataset] = pivot
    
    # Combine all pivots
    if not pivots:
        return None
        
    combined = pd.concat(pivots.values(), axis=1)
    
    # Add average column
    sts_columns = [col for col in combined.columns if col.startswith('STS')]
    if sts_columns:
        combined['AVG_STS'] = combined[sts_columns].mean(axis=1)
    
    # Sort by average score
    if 'AVG_STS' in combined.columns:
        combined = combined.sort_values('AVG_STS', ascending=False)
    
    return combined

def extract_config_details(df):
    """Extract configuration details from the config_id for a cleaner view."""
    if 'config_id' not in df.columns:
        return df
        
    # Extract components from config_id
    df['experiment'] = df['config_id'].str.extract(r'^([^_]+)')
    df['method'] = df['config_id'].str.extract(r'_([^_]+)_')
    df['gen_model'] = df['config_id'].str.extract(r'_[^_]+_([^_]+)_')
    df['embedding_model'] = df['config_id'].str.extract(r'_[^_]+_[^_]+_([^_]+)_')
    df['k_value'] = df['config_id'].str.extract(r'k(\d+)_')
    df['seed'] = df['config_id'].str.extract(r's(\d+)$')
    
    return df

def create_visualizations(df, output_dir):
    """Create visualizations to help understand the results."""
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a combined table focused on the spearman metric
    df_for_viz = df[df['dataset'].isin(STS_DATASETS)].copy()
    
    # Focus on spearman
    metric = 'spearman'
    metric_df = df_for_viz[['config_id', 'experiment', 'method', 'gen_model', 'embedding_model', 
                      'k_value', 'seed', 'dataset', metric]].copy()
    
    # Create a pivot table for the heatmap
    pivot = metric_df.pivot_table(
        index=['experiment', 'method', 'gen_model', 'embedding_model', 'k_value'],
        columns='dataset',
        values=metric,
        aggfunc='mean'  # Average if multiple seeds
    )
    
    # 1. Heatmap of all configurations across datasets
    plt.figure(figsize=(16, max(8, len(pivot) // 2)))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title(f'Comparison of Configurations across STS Datasets ({metric})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sts_heatmap_{metric}.png'), dpi=300)
    plt.close()
    
    # 2. Bar chart of average performance
    # Calculate average scores
    avg_scores = pivot.mean(axis=1).sort_values(ascending=False)
    
    # Create a more readable index
    avg_scores.index = [f"{exp}_{meth}_k{k}" 
                      for exp, meth, gen, emb, k in avg_scores.index]
    
    # Plot top 10
    plt.figure(figsize=(14, 8))
    avg_scores.head(10).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Top 10 Configurations by Average STS Score ({metric})')
    plt.ylabel(f'Average {metric} Score')
    plt.axhline(y=avg_scores.head(10).mean(), color='red', linestyle='--', 
                label=f'Mean of Top 10: {avg_scores.head(10).mean():.3f}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top10_configs_{metric}.png'), dpi=300)
    plt.close()
    
    # 3. Bar chart for each dataset showing the best configuration
    for dataset in STS_DATASETS:
        if dataset not in pivot.columns:
            continue
            
        # Sort configurations by performance on this dataset
        dataset_scores = pivot[dataset].sort_values(ascending=False)
        
        # Create a more readable index for top 5
        dataset_scores = dataset_scores.head(5)
        dataset_scores.index = [f"{exp}_{meth}_k{k}" 
                              for exp, meth, gen, emb, k in dataset_scores.index]
        
        # Plot top 5 for this dataset
        plt.figure(figsize=(12, 6))
        dataset_scores.plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.title(f'Top 5 Configurations for {dataset} ({metric})')
        plt.ylabel(f'{metric} Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset}_top5_{metric}.png'), dpi=300)
        plt.close()
    
    # 4. Comparison of different k values for the best experiment/method
    if 'k_value' in metric_df.columns:
        # Find the best experiment/method combination
        best_combo = pivot.mean(axis=1).idxmax()
        if isinstance(best_combo, tuple) and len(best_combo) >= 4:
            best_exp, best_method = best_combo[0], best_combo[1]
            
            # Filter to just this experiment/method
            k_compare_df = metric_df[(metric_df['experiment'] == best_exp) & 
                                   (metric_df['method'] == best_method)]
            
            if not k_compare_df.empty:
                # Create pivot table comparing k values
                k_pivot = k_compare_df.pivot_table(
                    index='k_value',
                    columns='dataset',
                    values=metric,
                    aggfunc='mean'
                )
                
                if not k_pivot.empty:
                    # Convert index to numeric for proper ordering
                    k_pivot.index = pd.to_numeric(k_pivot.index)
                    k_pivot = k_pivot.sort_index()
                    
                    # Add average column
                    k_pivot['AVG'] = k_pivot.mean(axis=1)
                    
                    # Plot
                    plt.figure(figsize=(14, 8))
                    for col in k_pivot.columns:
                        plt.plot(k_pivot.index, k_pivot[col], marker='o', label=col)
                    
                    plt.title(f'Effect of k Value for {best_exp}_{best_method} ({metric})')
                    plt.xlabel('k Value')
                    plt.ylabel(f'{metric} Score')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'k_value_effect_{metric}.png'), dpi=300)
                    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Gather and analyze STS benchmark results')
    parser.add_argument('--output_dir', default=BASE_DIR, help='Directory to save results')
    parser.add_argument('--create_viz', action='store_true', help='Create visualizations')
    parser.add_argument('--viz_only', action='store_true', help='Only create visualizations, using existing CSVs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up output paths
    detailed_csv_path = os.path.join(args.output_dir, "sts_results_detailed.csv")
    pivot_csv_path = os.path.join(args.output_dir, "sts_results_summary.csv")
    spearman_csv_path = os.path.join(args.output_dir, "sts_results_spearman.csv")
    combined_csv_path = os.path.join(args.output_dir, "sts_results_combined.csv")
    viz_dir = os.path.join(args.output_dir, "sts_visualizations")
    
    if args.viz_only:
        # Only create visualizations using existing CSV files
        if os.path.exists(detailed_csv_path):
            print(f"Loading existing detailed results from: {detailed_csv_path}")
            results_df = pd.read_csv(detailed_csv_path)
            
            if args.create_viz:
                print("Creating visualizations...")
                create_visualizations(results_df, viz_dir)
        else:
            print(f"Error: {detailed_csv_path} does not exist. Please run without --viz_only first.")
        return
    
    print("Gathering STS benchmark results...")
    results = gather_all_results()
    
    if not results:
        print("No results found!")
        return
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Create detailed CSV with all raw results
    results_df.to_csv(detailed_csv_path, index=False)
    print(f"Detailed results saved to: {detailed_csv_path}")
    
    # Create pivot tables
    pivot_tables = create_pivot_tables(results_df)
    
    # Create a single CSV with multiple tables
    with open(pivot_csv_path, 'w') as f:
        for table_name, pivot_df in pivot_tables.items():
            f.write(f"\n\n{table_name}\n")
            pivot_df.to_csv(f)
    
    print(f"Summary results saved to: {pivot_csv_path}")
    
    # Additional CSV with key metrics only (spearman scores)
    spearman_pivot = pivot_tables.get("spearman_spearman")
    if spearman_pivot is not None:
        spearman_pivot.to_csv(spearman_csv_path)
        print(f"Spearman results saved to: {spearman_csv_path}")
    
    # Create a combined table with the main metric (spearman) for all datasets
    combined_table = create_combined_table(results_df)
    if combined_table is not None:
        # Extract configuration details for a cleaner view
        combined_table = extract_config_details(combined_table)
        
        # Save the combined table
        combined_table.to_csv(combined_csv_path)
        print(f"Combined results saved to: {combined_csv_path}")
        
        # Print top 5 configurations by average score
        if 'AVG_STS' in combined_table.columns and len(combined_table) > 0:
            print("\nTop 5 configurations by average STS score:")
            top5 = combined_table.sort_values('AVG_STS', ascending=False).head(5)
            print(top5[['AVG_STS', 'experiment', 'method', 'gen_model', 'embedding_model', 'k_value']])
    
    # Create visualizations if requested
    if args.create_viz:
        print("Creating visualizations...")
        create_visualizations(results_df, viz_dir)

if __name__ == "__main__":
    main() 