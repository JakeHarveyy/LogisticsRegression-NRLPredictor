#!/usr/bin/env python3
"""
NRL Betting Data Preprocessing Pipeline - Main Execution Script

This script runs the complete NRL data preprocessing pipeline using DataFrames only,
saving CSV files only at the final step for maximum efficiency.

Usage:
    python run_pipeline.py

The pipeline will process all data in-memory and output:
- ../nrl_matches_final_model_ready.csv (main ML dataset)
- ../nrl_team_stats_final_complete.csv (complete team features)
- ../nrl_core_features_list.txt (feature reference)

Author: NRL Betting Model Team
Date: July 2025
"""

import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all required functions from the feature engineering module
from featureEngineering.featureEngineering import (
    load_and_clean_nrl_data,
    preview_data,
    create_team_level_stats,
    preview_team_stats,
    calculate_rolling_features,
    analyze_rolling_features,
    calculate_elo_ratings,
    calculate_rest_days,
    calculate_travel_distance,
    assemble_final_model_ready_dataframe,
    final_dataset_analysis
)

def run_nrl_preprocessing_pipeline():
    """
    Execute the complete NRL betting data preprocessing pipeline.
    
    This function coordinates all preprocessing steps:
    1. Data loading and cleaning
    2. Team-level transformation
    3. Rolling features calculation
    4. Strength and contextual features
    5. Final model-ready assembly
    6. CSV output generation
    
    Returns:
        tuple: (df_final, team_stats_final, core_features) - The main outputs
    """
    
    print(" Starting NRL Betting Data Preprocessing Pipeline...")
    print(" Using in-memory DataFrames for efficiency")
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load and clean the data
        print("\n Step 1: Foundational Cleaning & Setup - STARTING...")
        df = load_and_clean_nrl_data()
        preview_data(df)
        print(" Step 1: Foundational Cleaning & Setup - COMPLETE!")
        
        # Step 2: Create team-level stats DataFrame
        print("\n Step 2: Team-Level Stats DataFrame - STARTING...")
        team_stats_df = create_team_level_stats(df)
        
        # Preview team stats with a specific team example
        unique_teams = sorted(team_stats_df['team_name'].unique())
        example_team = unique_teams[0] if unique_teams else None
        
        if example_team:
            preview_team_stats(team_stats_df, team_name=example_team, n_rows=8)
        
        preview_team_stats(team_stats_df, n_rows=10)
        print(" Step 2: Team-Level Stats DataFrame - COMPLETE!")
        
        # Step 3: Calculate rolling features and form indicators
        print("\n Step 3: Form & Rolling Features - STARTING...")
        team_stats_enhanced = calculate_rolling_features(team_stats_df)
        
        # Analyze the rolling features with a sample team
        sample_analysis = analyze_rolling_features(team_stats_enhanced)
        print(" Step 3: Form & Rolling Features - COMPLETE!")
        
        # Step 4: Calculate strength and contextual features
        print("\n Step 4: Strength & Contextual Features - STARTING...")
        
        # Step 4a: Calculate Elo ratings
        team_stats_with_elo = calculate_elo_ratings(team_stats_enhanced)
        
        # Step 4b: Calculate rest days
        team_stats_with_rest = calculate_rest_days(team_stats_with_elo)
        
        # Step 4c: Calculate travel distance
        team_stats_final = calculate_travel_distance(team_stats_with_rest)
        print(" Step 4: Strength & Contextual Features - COMPLETE!")
        
        # Step 5: Assemble the Final Model-Ready DataFrame
        print("\n Step 5: Final Model-Ready DataFrame - STARTING...")
        df_final, core_features = assemble_final_model_ready_dataframe(df, team_stats_final)
        print(" Step 5: Final Model-Ready DataFrame - COMPLETE!")
        
        # Return the main outputs for potential further use
        return df_final, team_stats_final, core_features
        
    except Exception as e:
        print(f"\n ERROR: Pipeline failed at step: {e}")
        print(f"Error details: {str(e)}")
        raise e

def save_final_outputs(df_final, team_stats_final, core_features, save_intermediate=False):
    """
    Save all final outputs to CSV files.
    
    Args:
        df_final (pd.DataFrame): Final model-ready dataset
        team_stats_final (pd.DataFrame): Complete team stats dataset
        core_features (list): List of core model features
        save_intermediate (bool): Whether to save intermediate datasets
    """
    
    print("\n SAVING FINAL DATASETS TO CSV...")
    
    try:
        # Save the main model-ready dataset
        final_match_output = 'data/nrl_matches_final_model_ready.csv'
        df_final.to_csv(final_match_output, index=False)
        print(f"✓ Final model-ready dataset saved to: {final_match_output}")
        
        # Save the complete team stats dataset
        final_team_output = 'data/nrl_team_stats_final_complete.csv'
        team_stats_final.to_csv(final_team_output, index=False)
        print(f"✓ Final team stats with all features saved to: {final_team_output}")
        
        # Save a feature list for reference
        feature_list_output = 'data/nrl_core_features_list.txt'
        with open(feature_list_output, 'w') as f:
            f.write("NRL BETTING MODEL - CORE FEATURES LIST\n")
            f.write("="*50 + "\n\n")
            f.write("TARGET VARIABLE:\n")
            f.write("- Home_Win\n\n")
            f.write("CORE MODEL FEATURES (X):\n")
            for i, feature in enumerate(core_features, 1):
                f.write(f"{i:2d}. {feature}\n")
            f.write(f"\nTotal Features: {len(core_features)}\n")
        
        print(f"✓ Core features list saved to: {feature_list_output}")
        
        # Optional: Save intermediate datasets for debugging/analysis
        if save_intermediate:
            print("\n SAVING INTERMEDIATE DATASETS FOR ANALYSIS...")
            
            # Note: We need to import these from the main pipeline if intermediate saving is needed
            # This would require modifying the pipeline to return intermediate results
            print("  Intermediate dataset saving requires pipeline modification")
            print("   Set save_intermediate=False or modify pipeline to return intermediate DataFrames")
        
        print(f"\n✅ ALL OUTPUTS SAVED SUCCESSFULLY!")
        print(f"   📁 Main dataset: {final_match_output}")
        print(f"   📁 Team stats: {final_team_output}")
        print(f"   📁 Feature list: {feature_list_output}")
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to save outputs: {e}")
        raise e

def main():
    """
    Main execution function - runs the complete pipeline and saves outputs.
    """
    
    start_time = datetime.now()
    
    try:
        # Run the preprocessing pipeline
        df_final, team_stats_final, core_features = run_nrl_preprocessing_pipeline()
        
        # Save all final outputs
        save_intermediate = False  # Change to True if you want intermediate files
        save_final_outputs(df_final, team_stats_final, core_features, save_intermediate)
        
        # Run final comprehensive analysis
        print("\n FINAL ANALYSIS - STARTING...")
        final_dataset_analysis()
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        print(f"\n PIPELINE COMPLETE!")
        print(f"     Total execution time: {execution_time}")
        print(f"    Final dataset shape: {df_final.shape}")
        print(f"    Core features: {len(core_features)}")
        print(f"    All data processed in-memory with final CSV outputs only.")
        
        return True
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        print(f"\n PIPELINE FAILED!")
        print(f"   Execution time before failure: {execution_time}")
        print(f"   Error: {str(e)}")
        
        return False

if __name__ == "__main__":
    """
    Script entry point - run the complete NRL preprocessing pipeline.
    """
    # Execute the main pipeline
    success = main()
    
    if success:
        print("\n SUCCESS: Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n FAILURE: Pipeline encountered errors!")
        sys.exit(1)
