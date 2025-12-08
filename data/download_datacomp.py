#!/usr/bin/env python3
import os
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

# Configuration
REPO_ID = "UCSC-VLAA/Recap-DataComp-1B"
LOCAL_DIR = "./data/recap_subset"  # Where to save the files
PERCENTAGE = 0.01             # 1% of the dataset

def download_subset():
    print(f"Fetching file list from {REPO_ID}...")
    
    # 1. Get all files in the repository
    all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    # 2. Filter for only the training parquet files
    # The files are usually named like "data/train-00000-of-08192.parquet"
    parquet_files = [f for f in all_files if f.endswith(".parquet") and "train" in f]
    
    # 3. Calculate how many files represent 1%
    total_files = len(parquet_files)
    subset_count = int(total_files * PERCENTAGE)
    
    if subset_count == 0:
        subset_count = 1 # Always download at least one
        
    subset_files = parquet_files[:subset_count]
    
    print(f"Found {total_files} total shards.")
    print(f"Downloading {subset_count} shards (~{PERCENTAGE*100}%) to '{LOCAL_DIR}'...")
    
    # 4. Download the files
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    for filename in tqdm(subset_files, desc="Downloading Parquet Files"):
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False # Set to True if you want to save space and use cache
        )
        
    print("\n✅ Download Complete!")
    print(f"You now have {subset_count} parquet files containing approx 10M+ captions.")
    print("You can load them using pandas: df = pd.read_parquet('recap_subset/data/...')")

if __name__ == "__main__":
    download_subset()
