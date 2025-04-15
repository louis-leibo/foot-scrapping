import pandas as pd
import glob
import os

def check_duplicates():
    # Find the most recent Premier League complete dataset
    files = glob.glob('data/premier_league_complete_*.csv')
    if not files:
        print("No Premier League complete dataset found")
        return
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Analyzing file: {latest_file}")
    
    # Read the dataset
    df = pd.read_csv(latest_file)
    print(f"\nTotal rows: {len(df)}")
    
    # Check for duplicates based on all columns
    duplicates = df[df.duplicated(keep=False)]
    if len(duplicates) > 0:
        print(f"\nFound {len(duplicates)} duplicate rows")
        print("\nSample of duplicate rows:")
        print(duplicates.head())
    else:
        print("\nNo duplicate rows found")
    
    # Check for duplicates based on key identifying columns
    key_columns = ['nation', 'pos', 'age', 'born', 'Season']
    duplicates_by_key = df[df.duplicated(subset=key_columns, keep=False)]
    if len(duplicates_by_key) > 0:
        print(f"\nFound {len(duplicates_by_key)} rows with duplicate key combinations")
        print("\nSample of rows with duplicate keys:")
        print(duplicates_by_key[key_columns].head())
    else:
        print("\nNo duplicate key combinations found")

if __name__ == "__main__":
    check_duplicates() 