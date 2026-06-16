# src/explainability.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap

def run_shap_analysis():
    # Ensure a directory exists to save our visualization screenshots
    os.makedirs('notebooks/screenshots', exist_ok=True)
    
    print("\n[SHAP] Loading models and evaluation data arrays...")
    # Load E-commerce assets
    rf_eco = joblib.load('models/random_forest_ecommerce.pkl')
    X_test_eco = pd.read_csv('data/processed/X_test_ecommerce.csv')
    
    # Load Credit Card assets
    rf_cc = joblib.load('models/random_forest_credit.pkl')
    X_test_cc = pd.read_csv('data/processed/X_test_credit.csv')
    
    # Use a background sample to speed up tree explainer calculations safely
    eco_sample = X_test_eco.sample(100, random_state=42)
    cc_sample = X_test_cc.sample(100, random_state=42)
    
    # 1. E-commerce SHAP Analysis
    print("[SHAP] Computing values for E-commerce Random Forest...")
    explainer_eco = shap.TreeExplainer(rf_eco)
    shap_values_eco = explainer_eco.shap_values(eco_sample)
    
    # Handle SHAP multi-class output indexing safely (binary classification index 1 for fraud)
    eco_vals = shap_values_eco[1] if isinstance(shap_values_eco, list) else shap_values_eco
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(eco_vals, eco_sample, show=False)
    plt.title("SHAP Feature Impact Summary: E-commerce Fraud Model", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('notebooks/screenshots/ecommerce_shap_summary.png', dpi=300)
    plt.close()
    
    # 2. Bank Credit Card SHAP Analysis
    print("[SHAP] Computing values for Bank Credit Card Random Forest...")
    explainer_cc = shap.TreeExplainer(rf_cc)
    shap_values_cc = explainer_cc.shap_values(cc_sample)
    
    cc_vals = shap_values_cc[1] if isinstance(shap_values_cc, list) else shap_values_cc
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(cc_vals, cc_sample, show=False)
    plt.title("SHAP Feature Impact Summary: Bank Credit Card Fraud Model", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('notebooks/screenshots/credit_shap_summary.png', dpi=300)
    plt.close()
    
    print("[SUCCESS] SHAP explainability visualizations exported safely to 'notebooks/screenshots/'.")

if __name__ == "__main__":
    run_shap_analysis()