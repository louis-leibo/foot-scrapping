import pandas as pd
import os

def merge_laliga_seasons():
    # File paths (ensure these files are in the same directory as this script or use full paths)
    file_names = [
        "Fbref-data/complete_dataset_ITA_Serie_A_2024.csv",
        "Fbref-data/complete_dataset_ITA_Serie_A_2023.csv",
        "Fbref-data/complete_dataset_ITA_Serie_A_2022.csv",
        "Fbref-data/complete_dataset_ITA_Serie_A_2021.csv",
        "Fbref-data/complete_dataset_ITA_Serie_A_2020.csv"
    ]

    # Load each file into a list of DataFrames
    dataframes = []
    for file in file_names:
        if os.path.exists(file):
            print(f"✅ Loading {file}")
            df = pd.read_csv(file)
            dataframes.append(df)
        else:
            print(f"⚠️ File not found: {file}")

    # Concatenate all DataFrames
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Sort by 'season' in descending order if the column exists
        if 'season' in merged_df.columns:
            merged_df = merged_df.sort_values(by='season', ascending=True).reset_index(drop=True)

        # Save the merged file
        output_file = "Fbref-data/merged_serieA_2020_2024.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"\n✅ Merged dataset saved to: {output_file}")
    else:
        print("❌ No files were loaded. Merge operation skipped.")

if __name__ == "__main__":
    merge_laliga_seasons()
