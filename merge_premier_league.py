import pandas as pd
import glob
import os
from datetime import datetime
import sys

def clean_column_names(df):
    # Remove duplicate column names by adding suffixes
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index] = [dup + f'.{n}' for n in range(sum(cols == dup))]
    df.columns = cols
    return df

def merge_premier_league_data():
    try:
        print("Starting Premier League data merge process...")
        
        # Get all Premier League intermediate files
        premier_files = glob.glob('data/intermediate_ENG-Premier League_*_*_2025*.csv')
        print(f"Found {len(premier_files)} files to process")
        
        # Dictionary to store DataFrames by season and stat type
        season_data = {}
        
        # Process each file
        for file in premier_files:
            try:
                # Extract season and stat type from filename
                filename = os.path.basename(file)
                # Split by underscore but keep Premier League together
                parts = filename.replace('intermediate_ENG-Premier League_', '').split('_')
                season = parts[0]
                stat_type = parts[1]
                
                print(f"\nProcessing {season} - {stat_type}")
                
                # Read the CSV file
                df = pd.read_csv(file)
                
                # Skip the second row which contains column descriptions
                df = df.iloc[1:]
                
                # Reset index after dropping row
                df = df.reset_index(drop=True)
                
                # Clean column names
                df = clean_column_names(df)
                
                # Initialize season dictionary if not exists
                if season not in season_data:
                    season_data[season] = {}
                
                # Store DataFrame by stat type
                season_data[season][stat_type] = df
                print(f"Successfully loaded {len(df)} rows")
                
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
        
        print(f"\nProcessed {len(season_data)} seasons")
        
        # List to store all player data
        all_player_data = []
        
        # Process each season
        for season, stat_dfs in season_data.items():
            try:
                print(f"\nProcessing season {season}")
                # Start with standard stats as base
                if 'standard' in stat_dfs:
                    base_df = stat_dfs['standard'].copy()
                    print(f"Base dataframe has {len(base_df)} rows")
                    
                    # Add season column
                    base_df['Season'] = season
                    
                    # Create a player identifier using available columns
                    base_df['player_id'] = base_df.apply(lambda x: f"{x['nation']}_{x['pos']}_{x['age']}_{x['born']}", axis=1)
                    
                    # Merge other stat types
                    for stat_type, df in stat_dfs.items():
                        if stat_type != 'standard':
                            print(f"Merging {stat_type} stats")
                            # Create the same player_id in the stat DataFrame
                            df['player_id'] = df.apply(lambda x: f"{x['nation']}_{x['pos']}_{x['age']}_{x['born']}", axis=1)
                            
                            # Merge on player_id
                            base_df = pd.merge(base_df, df, 
                                             on='player_id',
                                             how='left',
                                             suffixes=('', f'_{stat_type}'))
                            print(f"After merging {stat_type}: {len(base_df)} rows")
                    
                    # Drop the temporary player_id column
                    base_df = base_df.drop('player_id', axis=1)
                    
                    # Drop duplicate columns from merging
                    base_df = base_df.loc[:, ~base_df.columns.duplicated()]
                    
                    # Reset index to ensure player_id is a regular column
                    base_df = base_df.reset_index(drop=True)
                    
                    all_player_data.append(base_df)
                    print(f"Completed processing season {season}")
                else:
                    print(f"Warning: No standard stats found for season {season}")
                    
            except Exception as e:
                print(f"Error processing season {season}: {str(e)}")
                continue
        
        # Reset index for all player data before concatenation
        all_player_data = [df.reset_index(drop=True) for df in all_player_data]
        
        # Combine all seasons
        if all_player_data:
            print("\nCombining all seasons...")
            final_df = pd.concat(all_player_data, ignore_index=True)
            
            # Ensure player_id is included in the final DataFrame
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
            final_df = final_df.drop_duplicates(subset=['player_id'], keep='first')
            
            # Reset index for final DataFrame
            final_df = final_df.reset_index(drop=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'data/premier_league_complete_{timestamp}.csv'
            
            # Save to CSV
            final_df.to_csv(output_file, index=False)
            print(f"\nCreated consolidated Premier League dataset: {output_file}")
            print(f"Total rows: {len(final_df)}")
            print(f"Number of columns: {len(final_df.columns)}")
            print("\nFirst few column names:")
            print(', '.join(list(final_df.columns)[:10]))
        else:
            print("No data found to merge")
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    merge_premier_league_data() 