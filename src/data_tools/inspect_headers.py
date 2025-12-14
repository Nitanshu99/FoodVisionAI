"""
Parquet Header Inspector.
Reads the schema of every Parquet file in the database directory.
"""

import pandas as pd
from pathlib import Path

def inspect_parquet_headers():
    # Path to your Parquet Database
    # Adjusted based on your project structure: data/processed/parquet_db
    base_dir = Path("data/processed/parquet_db")
    
    print(f"--- Inspecting Parquet Files in {base_dir} ---\n")

    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return

    # Find all .parquet files (case insensitive)
    files = list(base_dir.glob("*.parquet"))
    
    if not files:
        print("❌ No .parquet files found.")
        return

    for file_path in sorted(files):
        try:
            # Read just the columns (pyarrow allows reading schema without loading data)
            # using pandas read_parquet is easiest for printing
            df = pd.read_parquet(file_path)
            
            print(f"📂 File: {file_path.name}")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}\n")
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"⚠️ Error reading {file_path.name}: {e}\n")

if __name__ == "__main__":
    inspect_parquet_headers()