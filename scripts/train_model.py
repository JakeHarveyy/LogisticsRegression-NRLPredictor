# scripts/train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
import numpy as np
import joblib
import config 

def train():
    print("--- Starting Model Training ---")
    # --- 1. Load and Prepare Data ---
    df = pd.read_csv(config.DATA_FILE)
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    X_sorted = pd.get_dummies(df_sorted[config.FEATURES_TO_KEEP], columns=['temperature_category'], drop_first=True)
    y_sorted = df_sorted[config.TARGET_COLUMN]

    X_sorted.fillna(0, inplace=True)

    # --- 2. Cross-validation (performanc estimation) ---
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS)
    lr_model = LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000, class_weight='balanced')

    print("--- Running cross-validation to estimate performance ---")

    # Store results for each fold
    cv_results = {
        'train_accuracy': [],
        'val_accuracy': [],
        'train_auc': [],
        'val_auc': [],
        'train_logloss': [],
        'val_logloss': []
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted), 1):
        # --- 2.1. Split data into raw (unscaled) folds ---
        X_train_fold_raw, X_val_fold_raw = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
        y_train_fold, y_val_fold = y_sorted.iloc[train_idx], y_sorted.iloc[val_idx]
        
        # --- 2.2. Scale data INSIDE the loop to prevent data leakage ---
        # The scaler is fit ONLY on the training data of the current fold
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_raw)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold_raw) 

        # --- 2.3. Train the model on the SCALED training data ---
        lr_model.fit(X_train_fold_scaled, y_train_fold)
        
        # --- 2.4. Generate predictions using the SCALED data ---
        train_pred_proba = lr_model.predict_proba(X_train_fold_scaled)[:, 1]
        val_pred_proba = lr_model.predict_proba(X_val_fold_scaled)[:, 1]
        train_pred = lr_model.predict(X_train_fold_scaled)
        val_pred = lr_model.predict(X_val_fold_scaled)
        
        # --- 2.5. Calculate and store metrics for this fold ---
        cv_results['train_accuracy'].append(accuracy_score(y_train_fold, train_pred))
        cv_results['val_accuracy'].append(accuracy_score(y_val_fold, val_pred))
        cv_results['train_auc'].append(roc_auc_score(y_train_fold, train_pred_proba))
        cv_results['val_auc'].append(roc_auc_score(y_val_fold, val_pred_proba))
        cv_results['train_logloss'].append(log_loss(y_train_fold, train_pred_proba))
        cv_results['val_logloss'].append(log_loss(y_val_fold, val_pred_proba))
        
        # --- 2.6. Print fold summary ---
        val_acc_fold = cv_results['val_accuracy'][-1]
        val_auc_fold = cv_results['val_auc'][-1]
        train_acc_fold = cv_results['train_accuracy'][-1]
        
        print(f"  Fold {fold}: Train Acc={train_acc_fold:.3f}, Val Acc={val_acc_fold:.3f}, Val AUC={val_auc_fold:.3f}")
    
    # --- 2.7: Cross-Validation Results Summary ---
    print("\n --- 2.7: Cross-Validation Results Summary ---")

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


    # --- 3. Train Final Model and Scaler on ALL data ---
    print("Training final model and scaler on ALL data")
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_sorted)

    final_model = LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000, class_weight='balanced', solver='lbfgs')
    final_model.fit(X_final_scaled, y_sorted)

    # --- 4. Feature Importance Analysis ---
    print("\n=== 4: Feature Importance Analysis ===")

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

    # --- 4. Save Model ---
    print("---4. Save Model ---")
    print(f"Saving model to {config.MODEL_FILE}")
    joblib.dump(final_model, config.MODEL_FILE)

    print(f"Saving scaler to {config.SCALER_FILE}")
    joblib.dump(final_scaler, config.SCALER_FILE)

    print(f"saving Important features to model/nrl_feature_importance_baseline.csv")
    feature_importance.to_csv('model/nrl_feature_importance_baseline.csv', index=False)

    print("--- Model Training Complete ---")

if __name__ == '__main__':
    train()
