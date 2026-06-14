# src/train.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc

def evaluate_model(model, X_test, y_test, dataset_name="Dataset", model_name="Model"):
    """Evaluates the model focusing strictly on highly skewed fraud metrics."""
    print(f"\n--- Evaluation: {model_name} on {dataset_name} ---")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Check if model supports predict_proba (like Logistic Regression and Random Forest)
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        auc_pr = auc(recall, precision)
    else:
        auc_pr = np.nan

    # Print out standard classification metrics
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Area Under Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")
    
    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "auc_pr": auc_pr,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

def train_and_save_models():
    os.makedirs('models', exist_ok=True)
    
    # ----------------------------------------------------
    # PHASE A: E-COMMERCE STREAM MODELING
    # ----------------------------------------------------
    print("\n[LOADING] Fetching Processed E-commerce Data...")
    X_train_eco = pd.read_csv('data/processed/X_train_ecommerce.csv')
    X_test_eco = pd.read_csv('data/processed/X_test_ecommerce.csv')
    y_train_eco = pd.read_csv('data/processed/y_train_ecommerce.csv').values.ravel()
    y_test_eco = pd.read_csv('data/processed/y_test_ecommerce.csv').values.ravel()
    
    print("Training E-commerce Baseline (Logistic Regression)...")
    lr_eco = LogisticRegression(max_iter=1000, random_state=42)
    lr_eco.fit(X_train_eco, y_train_eco)
    evaluate_model(lr_eco, X_test_eco, y_test_eco, "E-commerce", "Logistic Regression")
    
    print("Training E-commerce Ensemble (Random Forest)...")
    rf_eco = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_eco.fit(X_train_eco, y_train_eco)
    evaluate_model(rf_eco, X_test_eco, y_test_eco, "E-commerce", "Random Forest")
    
    # Save top e-commerce model
    joblib.dump(rf_eco, 'models/random_forest_ecommerce.pkl')
    
    # ----------------------------------------------------
    # PHASE B: BANK CREDIT CARD STREAM MODELING
    # ----------------------------------------------------
    print("\n[LOADING] Fetching Processed Bank Credit Card Data...")
    X_train_cc = pd.read_csv('data/processed/X_train_credit.csv')
    X_test_cc = pd.read_csv('data/processed/X_test_credit.csv')
    y_train_cc = pd.read_csv('data/processed/y_train_credit.csv').values.ravel()
    y_test_cc = pd.read_csv('data/processed/y_test_credit.csv').values.ravel()
    
    print("Training Bank Baseline (Logistic Regression)...")
    lr_cc = LogisticRegression(max_iter=1000, random_state=42)
    lr_cc.fit(X_train_cc, y_train_cc)
    evaluate_model(lr_cc, X_test_cc, y_test_cc, "Bank Credit Card", "Logistic Regression")
    
    print("Training Bank Ensemble (Random Forest)...")
    rf_cc = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_cc.fit(X_train_cc, y_train_cc)
    evaluate_model(rf_cc, X_test_cc, y_test_cc, "Bank Credit Card", "Random Forest")
    
    # Save top bank model
    joblib.dump(rf_cc, 'models/random_forest_credit.pkl')
    print("\n[SUCCESS] All estimators successfully optimized and saved in 'models/'.")

if __name__ == "__main__":
    train_and_save_models()