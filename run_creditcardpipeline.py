import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Step 1: Loading Bank Credit Card Dataset ---")
credit_card_path = 'data/raw/creditcard.csv'

if not os.path.exists(credit_card_path):
    print(f"Error: Please ensure '{credit_card_path}' is placed inside your directory.")
else:
    # Load the dataset first
    df_credit = pd.read_csv(credit_card_path)
    
    # NOW we drop duplicates safely
    print("Removing duplicates...")
    df_credit = df_credit.drop_duplicates()
    
    print("\n--- Step 2: Quantifying Class Imbalance (EDA Verification) ---")
    total_records = len(df_credit)
    class_counts = df_credit['Class'].value_counts()
    class_percentages = df_credit['Class'].value_counts(normalize=True) * 100
    
    print(f"Total Transactions: {total_records}")
    print(f"Legitimate Transactions (Class 0): {class_counts[0]} ({class_percentages[0]:.4f}%)")
    print(f"Fraudulent Transactions (Class 1): {class_counts[1]} ({class_percentages[1]:.4f}%)")
    
    print("\n--- Step 3: Isolating Target Vector and Feature Matrices ---")
    X = df_credit.drop(columns=['Class'])
    y = df_credit['Class']
    
    print("--- Step 4: Stratified Train-Test Splitting (Preventing Data Leakage) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("--- Step 5: Scaling Continuous Numerical Features (StandardScaler) ---")
    # Scaling 'Time' and 'Amount' columns based strictly on training distributions to prevent leakage
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    print("--- Step 6: Committing Processed Splits to Disk ---")
    X_train.to_csv('data/processed/X_train_credit.csv', index=False)
    X_test.to_csv('data/processed/X_test_credit.csv', index=False)
    y_train.to_csv('data/processed/y_train_credit.csv', index=False)
    y_test.to_csv('data/processed/y_test_credit.csv', index=False)
    
    print("\n[SUCCESS] Bank pipeline completed flawlessly! Split matrices saved in 'data/processed/'.")
    print(f"Train dimensions: {X_train.shape} | Test dimensions: {X_test.shape}")