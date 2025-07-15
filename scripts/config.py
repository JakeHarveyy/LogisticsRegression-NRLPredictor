# scripts/config.py

# File Paths
DATA_FILE = 'data/nrl_matches_final_model_ready.csv'
MODEL_FILE = 'model/nrl_baseline_logistic_model_v1.pkl'
SCALER_FILE = 'model/nrl_feature_scaler_v1.pkl'
IMPORTANCE_FILE = 'model/nrl_feature_importance_baseline_v1.csv'
EVALUATION_OUTPUT_FILE = 'model/betting_simulation_results.txt'

# Model Configuration
TARGET_COLUMN = 'Home_Win'
FEATURES_TO_KEEP = [
    # Strength
    'elo_diff',
    
    # Form - Rolling Margins
    'form_margin_diff_3', 'form_margin_diff_5', 'form_margin_diff_8',
    
    # Form - Rolling Win Rate
    'form_win_rate_diff_3', 'form_win_rate_diff_5', 'form_win_rate_diff_8',
    
    # Form - Rolling Points For
    'form_points_for_diff_3', 'form_points_for_diff_5', 'form_points_for_diff_8',
    
    # Form - Rolling Points Against
    'form_points_against_diff_3', 'form_points_against_diff_5', 'form_points_against_diff_8',
    
    # Streaks and Recency
    'winning_streak_diff', 'losing_streak_diff', 'games_since_win_diff', 'games_since_loss_diff', 'recent_wins_3_diff',
    
    # Context - Fatigue & Travel
    'home_rest_days', 
    'away_rest_days',
    'away_travel_distance_km',
    
    # Market Intelligence
    'home_implied_prob', 'away_implied_prob', 'market_spread',

    # Weather Features
    'temperature_c', 'wind_speed_kph', 'precipitation_mm', 'is_rainy', 'is_windy', 'temperature_category'
]
N_SPLITS = 5
RANDOM_STATE = 42

# Betting Simulation Configuration
BETTING_THRESHOLD = 0.52
AVERAGE_ODDS = 1.90