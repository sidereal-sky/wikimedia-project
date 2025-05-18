#!/bin/bash

# Wikipedia Analysis Dashboard Setup Script

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    echo "https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html"
    exit 1
fi

# Set environment variables to avoid Spark connection issues
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_LOCAL_HOSTNAME=localhost

# Create or update conda environment
echo "Creating/updating conda environment from environment.yaml..."
conda env update -f environment.yaml

# Activate the environment (this won't work in the script but instructions will be shown)
echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate wikimedia-project"
echo ""
echo "Then run the dashboard with:"
echo "python run_dashboard.py"
echo ""
echo "If you encounter dependency issues, you can install required packages with:"
echo "conda install -c conda-forge pandas plotly dash folium numpy"
echo ""

# Create required directories
mkdir -p chunks
mkdir -p spark-temp
mkdir -p assets
mkdir -p cache

echo "Created necessary directories."
echo ""
echo "Place Wikipedia XML dump files (*.xml.bz2) in the 'chunks' directory."
echo "" 