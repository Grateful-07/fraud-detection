# src/features.py
import pandas as pd

def engineer_ecommerce_features(df):
    """Engineers velocity, temporal, and categorical features for the e-commerce dataset."""
    df = df.copy()
    
    # Defensive programming: Ensure times are datetimes even if raw data is injected
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Temporal attributes
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Velocity attributes
    df['device_velocity'] = df['device_id'].map(df['device_id'].value_counts())
    df['ip_velocity'] = df['ip_address'].map(df['ip_address'].value_counts())
    
    # Categorical encoding
    top_countries = df['country'].value_counts().index[:20]
    df['country'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')
    df_final = pd.get_dummies(df, columns=['source', 'browser', 'sex', 'country'], drop_first=True)
    
    # Drop raw operational identifiers
    features_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    df_final.drop(columns=features_to_drop, inplace=True, errors='ignore')
    return df_final