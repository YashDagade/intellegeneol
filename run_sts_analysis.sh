#!/bin/bash

# Set the base directory
BASE_DIR="/usr/project/xtmp/yd211/Documents/IntelleGENeol"
cd "$BASE_DIR"

# Check if Python and necessary packages are available
python -c "import pandas, matplotlib, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python packages (pandas, matplotlib, seaborn) are not available."
    echo "Please install them using: pip install pandas matplotlib seaborn"
    exit 1
fi

# Parse command line arguments
VIZ_ONLY=0
CREATE_VIZ=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --viz-only)
            VIZ_ONLY=1
            CREATE_VIZ=1
            shift
            ;;
        --create-viz)
            CREATE_VIZ=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--viz-only] [--create-viz]"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python gather_sts_results.py"

if [ $VIZ_ONLY -eq 1 ]; then
    CMD="$CMD --viz_only"
fi

if [ $CREATE_VIZ -eq 1 ]; then
    CMD="$CMD --create_viz"
fi

# Run the analysis
echo "Running STS benchmark analysis..."
echo "$CMD"
eval "$CMD"

# Display a summary of where to find the results
echo ""
echo "===== STS Benchmark Analysis Results ====="
echo "Detailed CSV: $BASE_DIR/sts_results_detailed.csv"
echo "Summary CSV: $BASE_DIR/sts_results_summary.csv"
echo "Spearman CSV: $BASE_DIR/sts_results_spearman.csv"
echo "Combined CSV: $BASE_DIR/sts_results_combined.csv"

if [ $CREATE_VIZ -eq 1 ]; then
    echo "Visualizations: $BASE_DIR/sts_visualizations/"
fi

echo ""
echo "To view results in Excel or similar tool, copy the CSV files to your local machine."
echo "===== Analysis Complete =====" 