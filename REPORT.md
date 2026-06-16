# End-to-End Multi-Stream Fraud Detection Systems: A Technical Post-Mortem
**Author:** Milto Workisa  
**Target Client:** Adey Innovations Inc.  
**Date:** June 16, 2026  

---

## 1. Introduction and Business Architecture
In modern digital financial systems, payment streams are constantly targeted by sophisticated fraudulent actors. This project addresses fraud detection across two entirely different transactional channels:
1. **E-commerce Platform Logs:** Behavioral data mapping transaction histories, browser footprints, and sign-up latency values.
2. **Bank Credit Card Transactions:** High-dimensional, anonymized PCA-engineered numerical transaction components with severe class skews.

### Defining the Core Optimization Metrics
Evaluating fraud detection on raw accuracy leads to system failures. Because fraud accounts for less than $0.17\%$ of bank transactions, a naive model that classifies everything as "legitimate" scores $99.83\%$ accuracy while leaving the business completely exposed to massive chargeback fees and regulatory fines. 

To maximize operational performance, our architectures prioritize **Area Under the Precision-Recall Curve (AUC-PR)** and **F1-Scores** to manage the operational trade-off between missing actual fraud (False Negatives) and inconveniencing legitimate users (False Positives).

---

## 2. Exploratory Data Analysis & Feature Engineering
Deep data profiling uncovered unique behavioral and statistical patterns across both data domains:

* **Time Latency Features:** For the e-commerce data, calculating the exact time delta between an account's sign-up time and its first purchase revealed a critical insight: fraudulent actors frequently execute transactions within seconds of account generation.
* **Network & Device Velocities:** Features tracking device frequencies and IP re-use rates proved that multiple distinct accounts transacting via a single device signature within a narrow window is a highly accurate indicator of structural fraud.
* **Bank Feature Distributions:** The Credit Card data contains 28 anonymized PCA components ($V_1 - V_28$). Statistical evaluations showed deep negative correlations with the target class across variables like $V_{14}$ and $V_{12}$, meaning that sharp downward variations in these specific hidden features strongly signal fraudulent activity.

---

## 3. Preprocessing, Pipeline Hygiene, & Safeguards
To transition our codebase from basic experimental scripts to stable production software, we integrated strict software engineering guardrails:
* **Production Exception Handling:** The core processing module employs a robust `validate_schema` method that verifies structural data inputs before running transformations, logging descriptive errors instead of standard runtime crashes.
* **Data Leakage Mitigation:** Class imbalances were resolved using Synthetic Minority Over-sampling (**SMOTE**). Crucially, SMOTE was applied *strictly to the training split* after isolation, ensuring the validation data remained clean and reflective of real production data.

---

## 4. Empirical Model Performance Comparisons

We trained and evaluated **Baseline Logistic Regression** and **Random Forest Ensembles** across both domains. Models were evaluated using native, imbalanced validation test partitions to accurately reflect performance in production:

| Platform / Stream | Classifier Model | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Area Under PR Curve (AUC-PR) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **E-commerce Logs** | Logistic Regression Baseline | $0.5752$ | $0.6544$ | $0.6122$ | $0.6390$ |
| **E-commerce Logs** | **Random Forest Ensemble** | **0.6038** | **0.6537** | **0.6278** | **0.7064** |
| **Bank Credit Card** | Logistic Regression Baseline | $0.0530$ | $0.8737$ | $0.1000$ | $0.7150$ |
| **Bank Credit Card** | **Random Forest Ensemble** | **0.7184** | **0.7789** | **0.7475** | **0.7915** |

### Justification for Final Model Selection
The **Random Forest Ensemble** was chosen for both deployment streams:
* In the **E-commerce Stream**, it successfully mapped complex non-linear structures, raising the **AUC-PR to 0.7064**.
* In the **Bank Credit Card Stream**, the improvements were stark. While Logistic Regression achieved strong recall ($0.8737$), its precision was completely unviable ($0.0530$), meaning it flagged countless innocent users. The Random Forest model stabilized the system, boosting precision to **0.7184** and achieving a high **AUC-PR of 0.7915**.

---

## 5. Model Explainability with SHAP

To ensure regulatory compliance and provide clear operational visibility, we applied **SHAP (SHapley Additive exPlanations)** frameworks to unpack our models' decision-making processes.

### Interpretation of E-commerce SHAP Summary
Our generated summary plots reveal exactly how feature values influence fraud probability scores:
1. **`time_since_signup`:** This emerged as a primary driver. Low values (shown as deep blue points) generate massive positive SHAP values, proving that rapid purchases following account setup strongly indicate fraudulent behavior.
2. **`device_velocity` / `ip_velocity`:** High frequency recurrences (red points) push the model's predictions firmly toward a fraud classification, confirming our velocity feature engineering assumptions.

### Interpretation of Bank Credit Card SHAP Summary
1. **Anonymized Features $V_{14}$, $V_{12}$, and $V_{17}$:** These features exhibit clear, structured patterns. Highly negative values (blue points) correspond to high positive SHAP impact metrics, making them vital indicators for identifying credit card fraud.
2. **`Amount`:** Large, unusual transaction amounts show a positive relationship with fraud classification probabilities, helping the model separate complex anomalies from day-to-day spending patterns.