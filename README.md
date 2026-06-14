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
3. Discussion of Completed Work and Modeling PerformanceFollowing our structural adjustments, we successfully built, validated, and evaluated two distinct machine learning frameworks (Baseline Logistic Regression and a high-capacity Random Forest Ensemble) across both operational streams.To maintain strict modeling hygiene, models were trained exclusively on data synthetic minority over-sampling (SMOTE) arrays, while evaluation was performed on pure, natively imbalanced test partitions to represent real-world deployment accuracy.Stream 1: E-commerce Transactions FrameworkLogistic Regression Baseline: Established a baseline, but struggled with highly non-linear feature signals (such as device velocities and geographical IP shifts).Precision (Class 1): $0.5752$Recall (Class 1): $0.6544$F1-Score (Class 1): $0.6122$AUC-PR: $0.6390$Random Forest Ensemble: Significantly minimized false alarms while tracking complex feature interactions.Precision (Class 1): $0.6038$Recall (Class 1): $0.6537$F1-Score (Class 1): $0.6278$AUC-PR: $0.7064$Stream 2: Bank Credit Card Transactions Framework (Extreme 0.17% Skew)Logistic Regression Baseline: While showing strong recall, it triggered a massive volume of false alerts due to the extreme class imbalance, rendering it unfeasible for real production operations.Precision (Class 1): $0.0530$Recall (Class 1): $0.8737$F1-Score (Class 1): $0.1000$AUC-PR: $0.7150$Random Forest Ensemble: Proved highly effective at navigating the anonymized PCA components ($V_1 - V_{28}$), dramatically clearing up false positives and bringing stability to the system.Precision (Class 1): $0.7184$Recall (Class 1): $0.7789$F1-Score (Class 1): $0.7475$AUC-PR: $0.7915$All best-performing models have been serialized and safely exported to the models/ directory for downstream deployment pipelines.4. Key Takeaways and Next Focus AreasDefensive Pipeline Safeguards: Addressed previous grader feedback by introducing an internal schema-validation layout (validate_schema) inside src/preprocessing.py, guaranteeing that incorrect column profiles trigger proactive system alerts.Advanced Boosting Architectures: The next phase will introduce XGBoost and LightGBM architectures paired with hyperparameter tuning via GridSearchCV to optimize precision-recall thresholds.Explainable AI (XAI): Integrating SHAP frameworks to make our models fully interpretable, detailing exactly how specific features drive fraud classifications.API Serving Infrastructure: Preparing to containerize our serialized .pkl estimators inside a lightweight FastAPI microservice wrapper for real-time transaction tracking.