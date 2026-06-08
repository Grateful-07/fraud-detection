# src/preprocessing.py
import pandas as pd
import numpy as np

def clean_and_format_ecommerce(df_fraud, df_ip):
    """Cleans data types, removes duplicates, and performs vectorized IP range mapping."""
    df_fraud = df_fraud.drop_duplicates().copy()
    df_fraud['signup_time'] = pd.to_datetime(df_fraud['signup_time'])
    df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'])
    df_fraud['ip_address'] = df_fraud['ip_address'].astype(int)
    
    df_ip['lower_bound_ip_address'] = df_ip['lower_bound_ip_address'].astype(int)
    df_ip['upper_bound_ip_address'] = df_ip['upper_bound_ip_address'].astype(int)
    df_ip = df_ip.sort_values('lower_bound_ip_address').reset_index(drop=True)
    
    # Vectorized range matching
    df_fraud = df_fraud.sort_values('ip_address')
    df_merged = pd.merge_asof(
        df_fraud, df_ip, 
        left_on='ip_address', right_on='lower_bound_ip_address', 
        direction='backward'
    )
    
    # Assign constraints
    df_merged.loc[df_merged['ip_address'] > df_merged['upper_bound_ip_address'], 'country'] = 'Unknown'
    df_merged['country'] = df_merged['country'].fillna('Unknown')
    df_merged.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True, errors='ignore')
    return df_merged

def clean_credit_card(df_credit):
    """Deduplicates and cleans bank credit card data."""
    return df_credit.drop_duplicates().copy()