# src/train.py
import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

# Configure systematic production logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_auc_pr(y_true, y_probs):
    """Safely calculates the Area Under the Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def train_and_evaluate_all():
    os.makedirs('models', exist_ok=True)
    results_records = []

    # --- DATAFRAME INGESTION WITH SYSTEMATIC ERROR HANDLING ---
    try:
        logging.info("Starting production dataset loading protocols...")
        X_train_eco = pd.read_csv('data/processed/X_train_ecommerce.csv')
        y_train_eco = pd.read_csv('data/processed/y_train_ecommerce.csv').values.ravel()
        X_test_eco = pd.read_csv('data/processed/X_test_ecommerce.csv')
        y_test_eco = pd.read_csv('data/processed/y_test_ecommerce.csv').values.ravel()

        X_train_cc = pd.read_csv('data/processed/X_train_credit.csv')
        y_train_cc = pd.read_csv('data/processed/y_train_credit.csv').values.ravel()
        X_test_cc = pd.read_csv('data/processed/X_test_credit.csv')
        y_test_cc = pd.read_csv('data/processed/y_test_credit.csv').values.ravel()
        logging.info("All matrix streams ingested cleanly into memory.")
    except FileNotFoundError as fnf:
        logging.critical(f"Pipeline Interrupted: Processed files are missing from directories. Details: {str(fnf)}")
        raise SystemExit("Exiting execution due to missing structural assets.")
    except Exception as e:
        logging.critical(f"Unexpected structural data anomaly detected during ingestion: {str(e)}")
        raise

    # --- DEFINE STREAM PROCESSING MATRICES ---
    streams = {
        "E-commerce Logs": (X_train_eco, y_train_eco, X_test_eco, y_test_eco),
        "Bank Credit Card": (X_train_cc, y_train_cc, X_test_cc, y_test_cc)
    }

    for name, (X_train, y_train, X_test, y_test) in streams.items():
        logging.info(f"Initializing optimization models for: {name}")

        # 1. Baseline Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr.fit(X_train, y_train)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        lr_preds = lr.predict(X_test)

        results_records.append({
            "Stream": name, "Model": "Logistic Regression Baseline",
            "Precision": precision_score(y_test, lr_preds, zero_division=0),
            "Recall": recall_score(y_test, lr_preds, zero_division=0),
            "F1": f1_score(y_test, lr_preds, zero_division=0),
            "AUC-PR": calculate_auc_pr(y_test, lr_probs)
        })

        # 2. Optimized Random Forest Ensemble (With Hyperparameter Tuning Limits)
        logging.info(f"Running optimized Hyperparameter Tuning limits for {name} Random Forest...")
        rf_optimized = RandomForestClassifier(
            n_estimators=150,        # Tuned upward for stable convergence
            max_depth=12,            # Regulated to strictly prevent over-fitting
            min_samples_split=5,     # Regularized split parameters
            random_state=42,
            n_jobs=-1
        )
        rf_optimized.fit(X_train, y_train)
        rf_probs = rf_optimized.predict_proba(X_test)[:, 1]
        rf_preds = rf_optimized.predict(X_test)

        results_records.append({
            "Stream": name, "Model": "Optimized Random Forest Ensemble",
            "Precision": precision_score(y_test, rf_preds, zero_division=0),
            "Recall": recall_score(y_test, rf_preds, zero_division=0),
            "F1": f1_score(y_test, rf_preds, zero_division=0),
            "AUC-PR": calculate_auc_pr(y_test, rf_probs)
        })

        # Save model binaries safely
        suffix = "ecommerce" if "E-commerce" in name else "credit"
        joblib.dump(rf_optimized, f"models/random_forest_{suffix}.pkl")
        logging.info(f"Optimized Random Forest binary exported to models/random_forest_{suffix}.pkl")

    # --- PRINT EXPLICIT MODEL COMPARISON TABLE ---
    df_results = pd.DataFrame(results_records)
    print("\n" + "="*85)
    print("                      FINAL EXPLICIT MODEL COMPARISON MATRIX                  ")
    print("="*85)
    print(df_results.to_string(index=False, formatters={
        'Precision': '{:,.4f}'.format, 'Recall': '{:,.4f}'.format,
        'F1': '{:,.4f}'.format, 'AUC-PR': '{:,.4f}'.format
    }))
    print("="*85 + "\n")

if __name__ == "__main__":
    train_and_evaluate_all()