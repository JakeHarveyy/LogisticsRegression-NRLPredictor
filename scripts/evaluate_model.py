# scripts/evaluate_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
import config # Import your new config file

def evaluate():
    print("--- starting Model Evaluation and Betting Simulation ---")

    # --- 1. Load Data and Trained Artifacts ---
    print("Loading data, model, and scaler...")
    df = pd.read_csv(config.DATA_FILE)
    model = joblib.load(config.MODEL_FILE)
    scaler = joblib.load(config.SCALER_FILE)

    # --- 2. Prepare Data for Prediction (mimic training prep) ---
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    X_sorted = pd.get_dummies(df_sorted[config.FEATURES_TO_KEEP], columns=['temperature_category'], drop_first=True)
    y_sorted = df_sorted[config.TARGET_COLUMN]

    X_sorted.fillna(0, inplace=True)

    # Use the LOADED scaler to transform the data
    X_scaled = scaler.transform(X_sorted)

    # --- 3. Generate Predictions ---
    final_predictions = model.predict(X_scaled)
    final_pred_proba = model.predict_proba(X_scaled)[:, 1]

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

    # --- 4 run the Betting Simulation ---
    print("\n--- Betting Performance Simulation (In-Sample) ---")

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

    print(" --- Evaluation Complete --- ")

if __name__ == '__main__':
    evaluate()