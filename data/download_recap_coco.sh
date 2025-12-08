#!/bin/bash

# Dataset URL: https://huggingface.co/datasets/UCSC-VLAA/Recap-COCO-30K

# Directory to save the dataset
DATA_DIR="data/recap_coco"
mkdir -p $DATA_DIR

echo "Downloading Recap-COCO-30K dataset to $DATA_DIR..."

# Download the parquet file
wget -O "$DATA_DIR/new_data.parquet" "https://huggingface.co/datasets/UCSC-VLAA/Recap-COCO-30K/resolve/main/new_data.parquet?download=true"

echo "Download complete."
