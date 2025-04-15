import pandas as pd
import numpy as np
from scraper import FBrefScraper
from config import AVAILABLE_LEAGUES, TARGET_SEASONS, STAT_TYPES
from datetime import datetime
import os

def create_ml_dataset():
    """
    Creates a comprehensive dataset for machine learning from FBref data.
    """
    # Initialize scraper
    scraper = FBrefScraper()
    
    # Collect all data
    print("Collecting data from FBref...")
    player_stats = scraper.scrape_all_leagues_seasons()
    
    # Merge all statistics
    print("Merging statistics...")
    merged_stats = scraper.merge_all_stats()
    
    if merged_stats.empty:
        print("No data collected!")
        return
    
    # Create derived features
    print("Creating derived features...")
    
    # Minutes per game
    merged_stats['minutes_per_game'] = merged_stats['minutes'] / merged_stats['games']
    
    # Goals per 90 minutes
    merged_stats['goals_per_90'] = (merged_stats['goals'] / merged_stats['minutes']) * 90
    
    # Assists per 90 minutes
    merged_stats['assists_per_90'] = (merged_stats['assists'] / merged_stats['minutes']) * 90
    
    # Pass completion rate
    merged_stats['pass_completion_rate'] = merged_stats['passes_completed'] / merged_stats['passes_attempted']
    
    # Shot accuracy
    merged_stats['shot_accuracy'] = merged_stats['shots_on_target'] / merged_stats['shots']
    
    # Tackle success rate
    merged_stats['tackle_success_rate'] = merged_stats['tackles_won'] / merged_stats['tackles']
    
    # Create a feature for player age (if available)
    if 'birth_date' in merged_stats.columns:
        merged_stats['age'] = pd.to_datetime('now').year - pd.to_datetime(merged_stats['birth_date']).dt.year
    
    # Create a feature for experience (years in the league)
    merged_stats['experience'] = merged_stats.groupby('player')['season'].rank()
    
    # Create a feature for team performance
    team_stats = merged_stats.groupby(['team', 'season']).agg({
        'goals': 'sum',
        'points': 'sum'
    }).reset_index()
    
    team_stats['team_goals_per_game'] = team_stats['goals'] / 38  # Assuming 38 games per season
    team_stats['team_points_per_game'] = team_stats['points'] / 38
    
    # Merge team performance back to player stats
    merged_stats = pd.merge(
        merged_stats,
        team_stats[['team', 'season', 'team_goals_per_game', 'team_points_per_game']],
        on=['team', 'season'],
        how='left'
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save the dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/ml_dataset_{timestamp}.csv"
    merged_stats.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Number of players: {merged_stats['player'].nunique()}")
    print(f"Number of seasons: {merged_stats['season'].nunique()}")
    print(f"Number of features: {len(merged_stats.columns)}")
    print("\nFeatures available:")
    for col in merged_stats.columns:
        print(f"- {col}")

if __name__ == "__main__":
    create_ml_dataset() 