# 10 Academy: Artificial Intelligence Mastery - Interim-2 Submission Report
**Project Title:** Multi-Stream Fraud Detection Engineering & Machine Learning Systems  
**Client:** Adey Innovations Inc.  
**Date:** June 14, 2026  

---

## 1. Understanding and Defining the Business Objective
In live payment streaming ecosystems, evaluating a classifier based on global raw accuracy leads to catastrophic operational failures. Because fraud occurrences represent extreme minority fragments ($9.36\%$ for e-commerce and an ultra-rare $0.17\%$ for credit cards), a naive system predicting that every transaction is "legitimate" would achieve over $99.8\%$ accuracy while completely missing real financial attacks.

Our model optimization explicitly targets the business trade-off between two core constraints:
* **The Cost of False Negatives (Missed Fraud):** Results in immediate financial loss, transaction chargebacks, and structural compliance penalties.
* **The Cost of False Positives (Insulting Legitimate Customers):** Causes unnecessary customer friction, lower user retention, and brand damage.

To balance these goals, our pipelines prioritize **Area Under the Precision-Recall Curve (AUC-PR)** and the **F1-Score** over standard classification accuracy.

---

## 2. Updated Project Directory Layout
```text
fraud-detection/
├── data/
│   ├── raw/                    # Source source arrays (Git-ignored)
│   └── processed/              # Cleaned, scaled, and SMOTE balanced sets
├── src/                        # Modular Core Code Architecture
│   ├── preprocessing.py        # Robust processing engine with schema validations
│   ├── features.py             # Velocity feature engineering pipelines
│   └── train.py                # Model training and optimization loops
├── notebooks/
│   ├── eda-fraud-data.ipynb    # E-commerce behavioral exploratory analyses
│   └── eda-credit-card.ipynb   # Credit Card PCA feature correlation analyses
├── models/                     # Target directory storing trained serialized binary files (.pkl)
├── tests/
│   └── test_pipeline.py        # Passing regression unit-tests
├── run_pipeline.py             # Data preparation pipeline orchestrator
├── run_modeling.py             # Model engineering pipeline runner
├── requirements.txt            # System dependencies (UTF-8 Encoded)
└── README.md                   # Core Technical Documentation (This Interim-2 Report)