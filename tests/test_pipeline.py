# tests/test_pipeline.py
import unittest
import pandas as pd
import numpy as np
from src.preprocessing import clean_credit_card
from src.features import engineer_ecommerce_features

class TestDataPipeline(unittest.TestCase):
    
    def test_credit_card_deduplication(self):
        # Create dummy data with explicit duplicates
        mock_data = pd.DataFrame({'Time': [1, 1, 2], 'Amount': [10.0, 10.0, 20.0], 'Class': [0, 0, 1]})
        cleaned_df = clean_credit_card(mock_data)
        # Should remove 1 row, leaving exactly 2 rows
        self.assertEqual(len(cleaned_df), 2)
        
    def test_ecommerce_feature_generation(self):
        # Verify that engineered parameters exist after processing
        mock_df = pd.DataFrame({
            'signup_time': ['2015-01-01 00:00:00'],
            'purchase_time': ['2015-01-01 02:00:00'],
            'purchase_value': [25],
            'device_id': ['DEV123'],
            'ip_address': [123456],
            'source': ['SEO'],
            'browser': ['Chrome'],
            'sex': ['M'],
            'country': ['United States'],
            'age': [30]
        })
        feature_df = engineer_ecommerce_features(mock_df)
        self.assertIn('time_since_signup', feature_df.columns)
        self.assertEqual(feature_df['time_since_signup'].iloc[0], 2.0)

if __name__ == '__main__':
    unittest.main()