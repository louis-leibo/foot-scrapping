import os
import glob
import csv
import pandas as pd

# Folder to search — current directory and subfolders
BASE_DIR = 'data'

# Pattern to match CSV filenames
pattern = os.path.join(BASE_DIR, '**', 'player_name_intermediate_*.csv')

# Find all matching CSV files (recursive)
files = glob.glob(pattern, recursive=True)

for file_path in files:
    print(f"Processing: {file_path}")

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header1 = next(reader)
            header2 = next(reader)
        except StopIteration:
            print(f"⚠️ Skipping {file_path} (not enough lines)")
            continue

    # Combine the two header rows
    combined_headers = [
        f"{h1.strip()} {h2.strip()}".strip() if h1.strip() and h2.strip() else h1.strip() or h2.strip()
        for h1, h2 in zip(header1, header2)
    ]

    # Read the rest of the data using pandas
    df = pd.read_csv(file_path, skiprows=2)

    # Set new headers
    df.columns = combined_headers

    # Overwrite the file with updated header
    df.to_csv(file_path, index=False)
    print(f"✅Updated: {file_path}")
