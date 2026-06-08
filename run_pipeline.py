# run_pipeline.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Import custom reusable modules
from src.preprocessing import clean_and_format_ecommerce, clean_credit_card
from src.features import engineer_ecommerce_features

def execute_ecommerce_pipeline():
    print("\n=== STARTING E-COMMERCE PIEPELINE ===")
    df_fraud_raw = pd.read_csv('data/raw/Fraud_Data.csv')
    df_ip_raw = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    
    # 1. Modular Processing
    df_cleaned = clean_and_format_ecommerce(df_fraud_raw, df_ip_raw)
    df_features = engineer_ecommerce_features(df_cleaned)
    
    X = df_features.drop(columns=['class'])
    y = df_features['class']
    
    # 2. Stratified Partitioning (No Leakage Design)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Explicit Numeric Scaling
    print("Applying StandardScaler to numerical attributes...")
    numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'device_velocity', 'ip_velocity']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # 4. Handling Class Imbalance via SMOTE (Training set only!)
    print(f"Class Distribution Before SMOTE: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Class Distribution After SMOTE: {dict(y_train_res.value_counts())}")
    
    # 5. Save out to disk
    X_train_res.to_csv('data/processed/X_train_ecommerce.csv', index=False)
    X_test.to_csv('data/processed/X_test_ecommerce.csv', index=False)
    y_train_res.to_csv('data/processed/y_train_ecommerce.csv', index=False)
    y_test.to_csv('data/processed/y_test_ecommerce.csv', index=False)
    print("[SUCCESS] E-commerce pipeline completed cleanly.")

def execute_bank_pipeline():
    print("\n=== STARTING BANK CREDIT CARD PIPELINE ===")
    cc_path = 'data/raw/creditcard.csv'
    if not os.path.exists(cc_path):
        print(f"Skipping: {cc_path} not found in data/raw/")
        return
        
    df_credit_raw = pd.read_csv(cc_path)
    df_cleaned = clean_credit_card(df_credit_raw)
    
    X = df_cleaned.drop(columns=['Class'])
    y = df_cleaned['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Explicit scaling on unscaled PCA raw parameters
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    # Apply SMOTE to the extreme 0.17% bank imbalance
    print(f"Credit Class Before SMOTE: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Credit Class After SMOTE: {dict(y_train_res.value_counts())}")
    
    X_train_res.to_csv('data/processed/X_train_credit.csv', index=False)
    X_test.to_csv('data/processed/X_test_credit.csv', index=False)
    y_train_res.to_csv('data/processed/y_train_credit.csv', index=False)
    y_test.to_csv('data/processed/y_test_credit.csv', index=False)
    print("[SUCCESS] Bank pipeline completed cleanly.")

if __name__ == "__main__":
    execute_ecommerce_pipeline()
    execute_bank_pipeline()