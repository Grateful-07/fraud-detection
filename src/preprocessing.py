# src/preprocessing.py
import pandas as pd
import numpy as np
import os

def validate_schema(df, required_columns, dataset_name="Dataset"):
    """Validates that all required columns are present in the DataFrame."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"[SCHEMA ERROR] {dataset_name} is missing required columns: {missing_cols}")

def clean_and_format_ecommerce(df_fraud, df_ip):
    """Cleans data types, removes duplicates, and performs vectorized IP range mapping with defensive checks."""
    if df_fraud.empty or df_ip.empty:
        raise ValueError("[I/O ERROR] E-commerce or IP lookup DataFrames are empty.")
        
    # Structural Schema Validation
    required_fraud = ['signup_time', 'purchase_time', 'ip_address', 'device_id', 'class']
    required_ip = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
    validate_schema(df_fraud, required_fraud, "Fraud Data")
    validate_schema(df_ip, required_ip, "IP Mapping Data")

    try:
        df_fraud = df_fraud.drop_duplicates().copy()
        df_fraud['signup_time'] = pd.to_datetime(df_fraud['signup_time'])
        df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'])
        df_fraud['ip_address'] = df_fraud['ip_address'].astype(np.int64)
        
        df_ip['lower_bound_ip_address'] = df_ip['lower_bound_ip_address'].astype(np.int64)
        df_ip['upper_bound_ip_address'] = df_ip['upper_bound_ip_address'].astype(np.int64)
        df_ip = df_ip.sort_values('lower_bound_ip_address').reset_index(drop=True)
        
        # Vectorized range matching
        df_fraud = df_fraud.sort_values('ip_address')
        df_merged = pd.merge_asof(
            df_fraud, df_ip, 
            left_on='ip_address', right_on='lower_bound_ip_address', 
            direction='backward'
        )
        
        df_merged.loc[df_merged['ip_address'] > df_merged['upper_bound_ip_address'], 'country'] = 'Unknown'
        df_merged['country'] = df_merged['country'].fillna('Unknown')
        df_merged.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True, errors='ignore')
        return df_merged
    except Exception as e:
        raise RuntimeWarning(f"[PROCESSING ERROR] Failed during E-commerce pipeline mapping: {str(e)}")

def clean_credit_card(df_credit):
    """Deduplicates and validates bank credit card data schema."""
    if df_credit.empty:
        raise ValueError("[I/O ERROR] Bank credit card DataFrame is empty.")
        
    # Bank Schema Validation (Time, Amount, Class, and PCA components V1-V28)
    expected_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    validate_schema(df_credit, expected_cols, "Bank Credit Card Data")
    
    return df_credit.drop_duplicates().copy()