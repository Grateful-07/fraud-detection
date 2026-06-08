import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("--- Step 1: Loading Datasets ---")
# Paths updated to reflect the required 10 Academy folder structure
df_fraud = pd.read_csv('data/raw/Fraud_Data.csv')
df_ip = pd.read_csv('data/raw/IpAddress_to_Country.csv')

print("--- Step 2: Cleaning and Formatting Types ---")
df_fraud['signup_time'] = pd.to_datetime(df_fraud['signup_time'])
df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'])
df_fraud['ip_address'] = df_fraud['ip_address'].astype(int)

df_ip['lower_bound_ip_address'] = df_ip['lower_bound_ip_address'].astype(int)
df_ip['upper_bound_ip_address'] = df_ip['upper_bound_ip_address'].astype(int)
df_ip = df_ip.sort_values('lower_bound_ip_address').reset_index(drop=True)

print("--- Step 3: Executing Range-Based Geolocation Lookup ---")
# Sorting by IP to leverage high-performance vectorized merge_asof matching
df_fraud = df_fraud.sort_values('ip_address')
df_merged = pd.merge_asof(
    df_fraud, 
    df_ip, 
    left_on='ip_address', 
    right_on='lower_bound_ip_address', 
    direction='backward'
)

# Assign 'Unknown' to IPs that fall outside the upper bound range constraints
df_merged.loc[df_merged['ip_address'] > df_merged['upper_bound_ip_address'], 'country'] = 'Unknown'
df_merged['country'] = df_merged['country'].fillna('Unknown')
df_merged.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)

print("--- Step 4: Engineering Advanced Behavioral & Temporal Features ---")
# 1. Time-since-signup in hours
df_merged['time_since_signup'] = (df_merged['purchase_time'] - df_merged['signup_time']).dt.total_seconds() / 3600.0

# 2. Extract purchase cyclic time components
df_merged['hour_of_day'] = df_merged['purchase_time'].dt.hour
df_merged['day_of_week'] = df_merged['purchase_time'].dt.dayofweek

# 3. Network & Terminal Velocity features (Multi-profile sharing patterns)
df_merged['device_velocity'] = df_merged['device_id'].map(df_merged['device_id'].value_counts())
df_merged['ip_velocity'] = df_merged['ip_address'].map(df_merged['ip_address'].value_counts())

print("--- Step 5: Encoding Categorical Variables ---")
# Keep top 20 countries and group the rest into 'Other' to protect feature matrix dimension scaling
top_countries = df_merged['country'].value_counts().index[:20]
df_merged['country'] = df_merged['country'].apply(lambda x: x if x in top_countries else 'Other')

categorical_cols = ['source', 'browser', 'sex', 'country']
df_final = pd.get_dummies(df_merged, columns=categorical_cols, drop_first=True)

# Separate predictive target and drop operational system identifiers
features_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
X = df_final.drop(columns=['class'] + features_to_drop)
y = df_final['class']

print("--- Step 6: Stratified Train-Test Partitioning (No-Leakage Design) ---")
# Splitting strictly before any resampling to ensure testing distribution stays authentic
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("--- Step 7: Saving Processed Sets to Disk ---")
X_train.to_csv('data/processed/X_train_ecommerce.csv', index=False)
X_test.to_csv('data/processed/X_test_ecommerce.csv', index=False)
y_train.to_csv('data/processed/y_train_ecommerce.csv', index=False)
y_test.to_csv('data/processed/y_test_ecommerce.csv', index=False)

print("\n[SUCCESS] Pipeline executed fully! Files saved in 'data/processed/' folder.")
print(f"Final training input dimension shape: {X_train.shape}")