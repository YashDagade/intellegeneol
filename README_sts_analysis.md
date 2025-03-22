# STS Benchmark Results Analysis

This tool helps you collect, analyze, and visualize STS benchmark results across different experimental configurations.

## Overview

The tool automatically searches through your result directories to find all STS benchmark results (STS12-STS22, STSBenchmark, SICK-R), extracts the metrics, and organizes them into easy-to-read CSV files. It can also create visualizations to help identify the best configurations.

## Files

- `gather_sts_results.py`: The main Python script that collects and processes the results
- `run_sts_analysis.sh`: A shell script to easily run the analysis

## Usage

### Basic Usage

To run the analysis and generate CSVs:

```
./run_sts_analysis.sh
```

### Generate Visualizations

To also generate visualizations:

```
./run_sts_analysis.sh --create-viz
```

### Only Generate Visualizations

If you've already run the analysis and just want to generate visualizations from the existing CSV files:

```
./run_sts_analysis.sh --viz-only
```

## Output Files

The tool generates several output files:

1. `sts_results_detailed.csv`: Contains all raw results with detailed configuration information
2. `sts_results_summary.csv`: Contains pivot tables for each metric across all datasets
3. `sts_results_spearman.csv`: Contains a focused view of spearman correlation scores
4. `sts_results_combined.csv`: Contains a combined table showing the main metrics for each configuration

If visualizations are enabled, they will be saved to the `sts_visualizations` directory and include:

1. Heatmaps comparing configurations across datasets
2. Bar charts showing the top 10 configurations by average score
3. Per-dataset bar charts showing the best configurations
4. Line charts showing the effect of k-value on performance

## How It Works

The tool:

1. Scans all result directories for JSON files containing STS benchmark results
2. Extracts configuration information from directory names (experiment, method, model, k-value, seed, etc.)
3. Parses the JSON files to extract performance metrics
4. Creates various summary tables and visualizations

## Example Workflow

1. Run multiple experiments with different configurations using your existing submission scripts
2. Once the experiments complete, run: `./run_sts_analysis.sh --create-viz`
3. Examine the generated CSV files to see which configurations performed best
4. Look at the visualizations in the `sts_visualizations` directory for deeper insights

## Tips

- The tool automatically detects configuration parameters from directory names
- The most important metrics are typically the spearman correlation scores
- The `sts_results_combined.csv` provides a good overall summary
- Check the visualizations to quickly identify trends across configurations 