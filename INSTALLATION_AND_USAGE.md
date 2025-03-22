# STS Analysis Tools - Installation and Usage Guide

## Files Created

1. `gather_sts_results.py` - The main Python script that collects and processes STS benchmark results
2. `run_sts_analysis.sh` - A shell script to run the analysis with various options
3. `README_sts_analysis.md` - Documentation about the tool's functionality

## Installation

Copy these files to your IntelleGENeol directory:

```bash
cd /usr/project/xtmp/yd211/Documents/IntelleGENeol
chmod +x run_sts_analysis.sh
```

## Usage

### Running Directly with Python

The simplest approach is to run the Python script directly:

```bash
cd /usr/project/xtmp/yd211/Documents/IntelleGENeol
python gather_sts_results.py
```

To enable visualizations:

```bash
python gather_sts_results.py --create_viz
```

### Using the Shell Script

If the shell script works correctly, you can use it with:

```bash
cd /usr/project/xtmp/yd211/Documents/IntelleGENeol
./run_sts_analysis.sh --create-viz
```

## What the Tool Does

This tool will:

1. Search through all your STS benchmark result files in the resultsTF directory
2. Extract metrics for STS12-18, STSBenchmark, SICK-R datasets
3. Create several CSV files with the results organized in different ways
4. Optionally create visualizations to help compare configurations

## Output Files

After running the tool, look for these files in your IntelleGENeol directory:

1. `sts_results_detailed.csv` - All raw results with detailed configuration information
2. `sts_results_summary.csv` - Pivot tables for each metric across all datasets
3. `sts_results_spearman.csv` - Focused view of spearman correlation scores
4. `sts_results_combined.csv` - Combined table showing the main metrics for each configuration

If visualizations are enabled, look for the `sts_visualizations` directory containing various plots.

## Troubleshooting

If you encounter any issues:

1. Make sure you have the necessary Python packages: pandas, matplotlib, seaborn
2. Check that the script has execution permissions (`chmod +x run_sts_analysis.sh`)
3. Try running the Python script directly instead of using the shell script
4. Verify that your result files are in the expected locations 