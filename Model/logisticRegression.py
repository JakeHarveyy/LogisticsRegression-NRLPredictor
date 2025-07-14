from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. Load model ready dataset ---
file_path = 'data/nrl_matches_final_model_ready.csv'  # This has the full 2009-2025 date range
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded master dataset: '{file_path}' with shape {df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find the file '{file_path}'. Please ensure it's in the correct directory.")
    exit()


# --- 2. Define the Target Variable (y) ---
target_column = 'Home_Win'
if target_column not in df.columns:
    print(f"Error: Target column '{target_column}' not found in the DataFrame.")
    exit()
    
y = df[target_column]
print(f"Target variable (y) set to '{target_column}'.")


# --- 3. Define training features (X) ---
# This is THE final, curated list of non-redundant, non-leaky features.
features_to_keep = [
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


print("\n=================================================")
print(" Ready to pass X and y to a model for training!")
print("=================================================")

# --- 6. BASELINE LOGISTIC REGRESSION MODEL TRAINING ---
print("\n" + "="*60)
print("BASELINE LOGISTIC REGRESSION MODEL TRAINING")
print("="*60)

# --- 6.1: Prepare Data for Time-Series Training ---
print("\n=== 6.1: Preparing Data for Time-Series Training ===")

# sort chronologically by date
df_sorted = df.sort_values('Date').reset_index(drop=True)
X_sorted = df_sorted[features_to_keep].copy()
y_sorted = df_sorted[target_column].copy()

# Fill any remaining missing values
X_sorted.fillna(0, inplace=True)

print(f"Data sorted chronologically: {X_sorted.shape[0]} matches")
print(f"Date range: {df_sorted['Date'].min()} to {df_sorted['Date'].max()}")

# # --- 6.2: Feature Scaling ---
# print("\n=== 6.2: Feature Scaling ===")

# # Logistic regression benefits from scaled features
X_sorted = pd.get_dummies(X_sorted, columns=['temperature_category'], drop_first=True)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_sorted)
# X_scaled = pd.DataFrame(X_scaled, columns=X_sorted.columns, index=X_sorted.index)

# print(f"Features scaled using StandardScaler")
# print(f"Feature means after scaling: {X_scaled.mean().round(3).tolist()[:5]}... (should be ~0)")
# print(f"Feature stds after scaling: {X_scaled.std().round(3).tolist()[:5]}... (should be ~1)")

# --- 6.3: Time-Series Cross-Validation Setup ---
print("\n=== 6.3: Time-Series Cross-Validation Setup ===")

# Use TimeSeriesSplit to respect temporal order (critical for betting models)
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"Using TimeSeriesSplit with {n_splits} splits")
print("This ensures we never train on future data to predict the past")

# --- 6.4: Train Baseline Logistic Regression ---
print("\n=== 6.4: Training Baseline Logistic Regression ===")

# Initialise model with balanced class weights (accounts for any home advantage bias)
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',  # Handles any class imbalance
    solver='lbfgs'
)

# Store results for each fold
cv_results = {
    'train_accuracy': [],
    'val_accuracy': [],
    'train_auc': [],
    'val_auc': [],
    'train_logloss': [],
    'val_logloss': []
}

print("Training model on each time-series fold...")

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted), 1):
    # --- 1. Split data into raw (unscaled) folds ---
    X_train_fold_raw, X_val_fold_raw = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
    y_train_fold, y_val_fold = y_sorted.iloc[train_idx], y_sorted.iloc[val_idx]
    
    # --- 2. Scale data INSIDE the loop to prevent data leakage ---
    # The scaler is fit ONLY on the training data of the current fold
    scaler_fold = StandardScaler()
    X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_raw)
    X_val_fold_scaled = scaler_fold.transform(X_val_fold_raw) # Use the same scaler to transform the validation data

    # --- 3. Train the model on the SCALED training data ---
    lr_model.fit(X_train_fold_scaled, y_train_fold)
    
    # --- 4. Generate predictions using the SCALED data ---
    # We only need the probabilities to calculate AUC and LogLoss
    train_pred_proba = lr_model.predict_proba(X_train_fold_scaled)[:, 1]
    val_pred_proba = lr_model.predict_proba(X_val_fold_scaled)[:, 1]

    # For accuracy, we need the class predictions
    train_pred = lr_model.predict(X_train_fold_scaled)
    val_pred = lr_model.predict(X_val_fold_scaled)
    
    # --- 5. Calculate and store metrics for this fold ---
    cv_results['train_accuracy'].append(accuracy_score(y_train_fold, train_pred))
    cv_results['val_accuracy'].append(accuracy_score(y_val_fold, val_pred))
    cv_results['train_auc'].append(roc_auc_score(y_train_fold, train_pred_proba))
    cv_results['val_auc'].append(roc_auc_score(y_val_fold, val_pred_proba))
    cv_results['train_logloss'].append(log_loss(y_train_fold, train_pred_proba))
    cv_results['val_logloss'].append(log_loss(y_val_fold, val_pred_proba))
    
    # --- 6. Print fold summary ---
    # We use the results directly from the dictionary for clarity
    val_acc_fold = cv_results['val_accuracy'][-1]
    val_auc_fold = cv_results['val_auc'][-1]
    train_acc_fold = cv_results['train_accuracy'][-1]
    
    print(f"  Fold {fold}: Train Acc={train_acc_fold:.3f}, Val Acc={val_acc_fold:.3f}, Val AUC={val_auc_fold:.3f}")

# --- 6.5: Cross-Validation Results Summary ---
print("\n=== 6.5: Cross-Validation Results Summary ===")

cv_summary = pd.DataFrame(cv_results)
print(" PERFORMANCE METRICS ACROSS ALL FOLDS:")
print(cv_summary.round(4))

print(f"\n AVERAGE PERFORMANCE:")
print(f"   Training Accuracy: {np.mean(cv_results['train_accuracy']):.3f} ± {np.std(cv_results['train_accuracy']):.3f}")
print(f"   Validation Accuracy: {np.mean(cv_results['val_accuracy']):.3f} ± {np.std(cv_results['val_accuracy']):.3f}")
print(f"   Validation AUC: {np.mean(cv_results['val_auc']):.3f} ± {np.std(cv_results['val_auc']):.3f}")
print(f"   Validation Log Loss: {np.mean(cv_results['val_logloss']):.3f} ± {np.std(cv_results['val_logloss']):.3f}")

# Check for overfitting
train_val_gap = np.mean(cv_results['train_accuracy']) - np.mean(cv_results['val_accuracy'])

print(f"\n OVERFITTING CHECK:")
print(f"   Train-Validation Accuracy Gap: {train_val_gap:.3f}")
if train_val_gap < 0.05:
    print("    Good: Low overfitting risk")
elif train_val_gap < 0.10:
    print("     Moderate: Some overfitting detected")
else:
    print("    High: Significant overfitting detected")

# --- 6.6: Train Final Model on All Data ---
print("\n=== 6.6: Training Final Model on All Data ===")

final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X_sorted)

# Train final model on all available data
final_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs'
)

final_model.fit(X_final_scaled, y_sorted)
print("Final model trained on complete dataset")

# --- 6.7: Feature Importance Analysis ---
print("\n=== 6.7: Feature Importance Analysis ===")

# Get feature coefficients (importance)
feature_importance = pd.DataFrame({
    'feature': X_sorted.columns,
    'coefficient': final_model.coef_[0],
    'abs_coefficient': np.abs(final_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(" TOP 10 MOST IMPORTANT FEATURES:")
print(feature_importance.head(10)[['feature', 'coefficient']].to_string(index=False))

print(f"\n FEATURE IMPORTANCE INSIGHTS:")
top_feature = feature_importance.iloc[0]
print(f"    Most Important: {top_feature['feature']} (coef: {top_feature['coefficient']:.3f})")

# Positive vs negative coefficients
positive_features = feature_importance[feature_importance['coefficient'] > 0]
negative_features = feature_importance[feature_importance['coefficient'] < 0]
print(f"   Features favoring HOME wins: {len(positive_features)}")
print(f"   Features favoring AWAY wins: {len(negative_features)}")

# --- 6.8: Final Predictions and Performance ---
print("\n=== 6.8: Final Model Performance ===")

final_predictions = final_model.predict(X_final_scaled)
final_pred_proba = final_model.predict_proba(X_final_scaled)[:, 1]

final_accuracy = accuracy_score(y_sorted, final_predictions)
final_auc = roc_auc_score(y_sorted, final_pred_proba)
final_logloss = log_loss(y_sorted, final_pred_proba)

print(f"FINAL MODEL PERFORMANCE:")
print(f"   Accuracy: {final_accuracy:.3f}")
print(f"   AUC-ROC: {final_auc:.3f}")
print(f"   Log Loss: {final_logloss:.3f}")

# Classification report
print(f"\n DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_sorted, final_predictions, target_names=['Away Win', 'Home Win']))

# --- 6.9: Betting Performance Simulation ---
print("\n=== 6.9: Betting Performance Simulation ===")

# Betting strategy with 0.52 threshold (accounts for ~4% bookmaker margin)
betting_threshold = 0.52

print(f" BETTING STRATEGY EXPLANATION:")
print(f"    Threshold: {betting_threshold} (accounts for bookmaker margin)")
print(f"    Bet on HOME when model probability > {betting_threshold}")
print(f"    Bet on AWAY when model probability < {1-betting_threshold}")
print(f"    No bet when probability between {1-betting_threshold:.2f} - {betting_threshold:.2f} (too uncertain)")

# Current strategy: HOME-only betting
print(f"\n CURRENT SIMULATION: HOME-ONLY BETTING")
home_bet_mask = final_pred_proba > betting_threshold

if home_bet_mask.sum() > 0:
    home_betting_accuracy = accuracy_score(y_sorted[home_bet_mask], final_predictions[home_bet_mask])
    home_total_bets = home_bet_mask.sum()
    home_total_wins = (final_predictions[home_bet_mask] == y_sorted[home_bet_mask]).sum()
    
    print(f" HOME-ONLY BETTING RESULTS:")
    print(f"   Total Bets Placed: {home_total_bets} ({home_total_bets/len(y_sorted)*100:.1f}% of matches)")
    print(f"   Betting Accuracy: {home_betting_accuracy:.3f}")
    print(f"   Total Wins: {home_total_wins}")
    print(f"   Win Rate: {home_total_wins/home_total_bets:.3f}")
    
    # Simple ROI calculation (assuming 1.90 odds average)
    home_simple_roi = (home_total_wins * 1.90 - home_total_bets) / home_total_bets
    print(f"   Estimated ROI: {home_simple_roi:.1%} (assuming 1.90 avg odds)")
else:
    print(" No HOME bets would be placed with current threshold")

# Enhanced strategy: Bet on BOTH home and away when confident
print(f"\n ENHANCED SIMULATION: BOTH HOME & AWAY BETTING")
away_bet_mask = final_pred_proba < (1 - betting_threshold)  # Away win probability > 0.52
both_bet_mask = home_bet_mask | away_bet_mask

if both_bet_mask.sum() > 0:
    # Determine bet predictions: 1 for home bets, 0 for away bets  
    bet_predictions = np.where(home_bet_mask, 1, 0)
    
    both_betting_accuracy = accuracy_score(y_sorted[both_bet_mask], bet_predictions[both_bet_mask])
    both_total_bets = both_bet_mask.sum()
    both_total_wins = (bet_predictions[both_bet_mask] == y_sorted[both_bet_mask]).sum()
    
    print(f"BOTH HOME & AWAY BETTING RESULTS:")
    print(f"   Total Bets Placed: {both_total_bets} ({both_total_bets/len(y_sorted)*100:.1f}% of matches)")
    print(f"   Home Bets: {home_bet_mask.sum()} | Away Bets: {away_bet_mask.sum()}")
    print(f"   Betting Accuracy: {both_betting_accuracy:.3f}")
    print(f"   Total Wins: {both_total_wins}")
    print(f"   Win Rate: {both_total_wins/both_total_bets:.3f}")
    
    # ROI calculation for both strategies
    both_simple_roi = (both_total_wins * 1.90 - both_total_bets) / both_total_bets
    print(f"   Estimated ROI: {both_simple_roi:.1%} (assuming 1.90 avg odds)")
    
    print(f"\n STRATEGY COMPARISON:")
    print(f"   Home-Only: {home_total_bets} bets, {home_simple_roi:.1%} ROI")
    print(f"   Both H&A: {both_total_bets} bets, {both_simple_roi:.1%} ROI")
else:
    print(" No bets would be placed with enhanced strategy")

print("\n" + "="*60)
print(" BASELINE LOGISTIC REGRESSION MODEL COMPLETE!")
print("="*60)

print(f"\n KEY TAKEAWAYS:")
print(f"   Validation Accuracy: {np.mean(cv_results['val_accuracy']):.1%}")
print(f"   Most Important Feature: {feature_importance.iloc[0]['feature']}")
print(f"   Model shows {'low' if train_val_gap < 0.05 else 'moderate' if train_val_gap < 0.10 else 'high'} overfitting")
print(f"   Ready for advanced model comparison (Random Forest, XGBoost, etc.)")

# Save model and results for future use
import joblib
joblib.dump(final_model, 'model/nrl_baseline_logistic_model.pkl')
joblib.dump(final_scaler, 'model/nrl_feature_scaler.pkl')
feature_importance.to_csv('model/nrl_feature_importance_baseline.csv', index=False)

print(f"\n SAVED:")
print(f"   Model: nrl_baseline_logistic_model.pkl")
print(f"   Scaler: nrl_feature_scaler.pkl") 
print(f"   Feature Importance: nrl_feature_importance_baseline.csv")