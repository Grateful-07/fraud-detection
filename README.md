# Multi-Stream Fraud Detection Systems: Data Engineering & Explainable AI
**Author:** Milto Workisa  
**Course:** 10 Academy: Artificial Intelligence Mastery - Final Project Delivery  
**Date:** June 16, 2026  

---

## 1. Project Overview
This repository contains a production-ready, modular machine learning system designed to ingest, validate, engineer, and evaluate transaction streams across e-commerce platforms and banking services for fraud detection. The system balances high class-imbalances using defensive splitting strategies, performs automated regression testing, and implements SHAP frameworks to provide full model explainability.

---

## 2. Professional Directory Architecture
```text
fraud-detection/
├── data/
│   ├── raw/                    # Original raw data vectors (Git-ignored)
│   └── processed/              # Validated, scaled, and SMOTE-partitioned data arrays
├── src/                        # Modular Production Source Layers
│   ├── preprocessing.py        # Robust ingestion module with strict schema validations
│   ├── features.py             # Behavioral velocity extraction engine
│   ├── train.py                # Model training, benchmarking, and export loops
│   └── explainability.py       # SHAP validation and visual asset exporter
├── notebooks/
│   ├── eda-fraud-data.ipynb    # E-commerce behavioral profile notebooks
│   ├── eda-credit-card.ipynb   # Credit card high-dimensional correlation notebooks
│   └── screenshots/            # Saved SHAP summary plot artifacts (.png)
├── models/                     # Production binary storage (.pkl format)
├── tests/
│   └── test_pipeline.py        # Automated testing suite
├── run_pipeline.py             # Master data pipeline runner
├── run_modeling.py             # Master model training loop runner
├── requirements.txt            # System dependencies (UTF-8 Encoded)
├── REPORT.md                   # Long-form End-to-End Technical Project Report
└── README.md                   # Main Project Landing Documentation (This File)