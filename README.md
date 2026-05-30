# 📉 Customer Churn Prediction — Telecom Company (Omni)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)
![SHAP](https://img.shields.io/badge/SHAP-0.42-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![UCC](https://img.shields.io/badge/UCC-MSc%20Business%20Analytics-darkgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**End-to-end ML pipeline · Data Generation → EDA → Feature Engineering → 
Outlier Detection → 6 ML Models → SHAP/LIME Interpretability · 
50,000 records · 35 features · UCC MSc Business Analytics**

> **Skills:** Python · Scikit-learn · XGBoost · SHAP · LIME · Pandas · Machine Learning · Feature Engineering · Outlier Detection

---

## Why This Matters

Predicting churn accurately is worth millions in retained revenue. 
This project goes beyond model accuracy — every prediction is explained 
using SHAP and LIME so a business stakeholder can act on it, not just 
trust it blindly.

---

## 📌 Project Overview

**Omni Company** is a large subscription-based telecom and digital services provider facing rising competitive pressure and customer attrition. This project builds an end-to-end **Customer Churn Prediction and Interpretability Framework** to identify at-risk customers before they leave — enabling targeted, proactive retention strategies.

**Core Business Question:**  
*Which customers are most likely to churn, and what are the main factors driving churn at Omni Company?*

**Dataset:** 50,000 customer records · 35 features · 22% churn rate (moderately imbalanced)  
**Primary Model:** Logistic Regression (highest recall for churners)  
**Best AUC Model:** XGBoost (ROC-AUC: 0.654)

---

## 🗂️ Repository Structure & How Files Connect

📁 Customer-Churn-Prediction/
│
├── 📓 Customer_Churn.ipynb                 ← MAIN FILE: Full ML pipeline
│   │   Reads: omni_churn_data_synthetic.csv
│   │   Outputs: model metrics, SHAP plots, LIME explanations, outlier reports
│
├── 📊 omni_churn_data_synthetic.csv        ← INPUT DATA
│   │   Used by: Customer_Churn.ipynb (Step 1 — Data Loading)
│   │   Structure mirrors original Omni dataset: 35 columns, 50K rows, 22% churn
│
├── 📄 Business_Problem_Statement.pdf       ← PROJECT BRIEF
│   │   Defines: business context, dataset schema, modelling tasks, evaluation rubric
│   │   Referenced in: README (overview), Notebook (Section 1 — Problem Definition)
│
├── 🔧 data_generator.py                    ← SYNTHETIC DATA GENERATOR
│   │   Generates: omni_churn_data_synthetic.csv from scratch
│   │   Run this first if CSV is not present
│
├── 📦 requirements.txt                     ← PYTHON DEPENDENCIES
│   │   Used by: Customer_Churn.ipynb (install before running)
│
└── 📜 LICENSE                              ← MIT License

### 🔗 Pipeline Flow

omni_churn_data_synthetic.csv
│
▼
[1] Data Loading & EDA          ← Notebook Section 1–2
│
▼
[2] Data Cleaning &             ← Notebook Section 3
Feature Engineering
(avg_charges_per_month,
tickets_per_month)
│
▼
[3] Outlier Detection           ← Notebook Section 4
(IQR + Z-Score + Domain Rules)
│
▼
[4] Train/Test Split            ← Notebook Section 5
& Feature Scaling
│
▼
[5] 6 ML Models Trained         ← Notebook Section 6
(LR · RF · DT · XGBoost
· ANN · Linear SVM)
│
▼
[6] Model Interpretability      ← Notebook Section 7
(SHAP · LIME · LOO ·
RF Feature Importance)
│
▼
[7] Results & Recommendations   ← Notebook Section 8

---

## 📊 Dataset — `omni_churn_data_synthetic.csv`

A synthetic dataset generated to mirror the original Omni Company dataset structure.

| Category | Features |
|---|---|
| Demographics | CustomerID, Age, Gender, IncomeTier, Region, Education, CityTier, CustomerSegment |
| Contract & Account | ContractLength, PlanType, PaymentMethod, TenureMonths, ContractAutoRenew, AutoPay, PaymentDelinquencyStatus, Paperless |
| Usage & Spend | MonthlyCharges, TotalCharges, LoginsLastMonth, RFMScore, UsageChangePct, CompetitorIndex |
| Engagement & Support | TicketsOpened, TicketsResolutionTime, SupportChannelPreferred, ComplaintCategory |
| Products & Bundles | FamilyPlan, AddOnBundle, DiscountType, PromoCodeUsed |
| Device | ReferralSource, ChannelPreferred, DeviceType, DeviceOS |
| **Target** | **Churn (0 = Retained, 1 = Churned)** |

> **Note:** The original dataset is proprietary to UCC / IS6052. The synthetic CSV replicates schema, data types, and statistical distributions (22% churn rate) for reproducibility.

---

## 🤖 Models Implemented

| # | Model | Accuracy | Recall (Churn) | ROC-AUC | Business Use |
|---|---|---|---|---|---|
| 1 | Logistic Regression | 0.679 | **0.472** | **0.677** | ✅ Primary — flags most at-risk customers |
| 2 | Random Forest | 0.775 | 0.004 | 0.622 | High precision — use for confirmed churners |
| 3 | Decision Tree | 0.718 | 0.460 | 0.630 | Explainable to non-technical stakeholders |
| 4 | XGBoost | 0.647 | 0.521 | 0.654 | ✅ Best AUC — SHAP interpretability benchmark |
| 5 | ANN (MLP) | 0.774 | 0.000 | 0.619 | Not suitable for this imbalanced dataset |
| 6 | Linear SVM | 0.834 | 0.417 | 0.741 | Strong overall but lower recall for churners |

**✅ Primary Model Selected: Logistic Regression** — highest recall for churners, meaning more at-risk customers are correctly identified before they leave. This is the critical business metric for Omni.

**XGBoost** used as benchmark model for feature importance and SHAP interpretability.

---

## 🔍 Interpretability Methods

| Method | Scope | Purpose |
|---|---|---|
| **SHAP** | Global + Local | Explains which features push predictions up/down for every customer |
| **LIME** | Local | Explains why one specific customer was flagged as high-risk |
| **LOO** | Global | Measures each feature's impact by removing it and re-evaluating |
| **RF Feature Importance** | Global | Gini-based importance from Random Forest model |

**Top 5 Churn Drivers (consistent across all 4 methods):**
1. Contract Length Month-To-Month — biggest AUC drop when removed (0.0354)
2. Payment Delinquency Status — second most important (0.0155)
3. Monthly Charges — third most important (0.0070)
4. AutoPay — fourth most important (0.0044)
5. Tenure Months — short tenure = unstable customer

---

## 🧹 Outlier Detection & Handling

Three-layered outlier detection on numerical features:

- **IQR Method** — flagged extreme values in Age, TenureMonths, MonthlyCharges, TotalCharges
- **Z-Score Thresholding** — identified statistical outliers (|z| > 3) across 8 numerical columns
- **Domain-Specific Rules** — TenureMonths > 150 is operationally impossible; Age > 90 is invalid

All outlier handling decisions documented in Notebook Section 3.5 with before/after comparison.

---

## What I'd Do With More Data
With actual post-churn return data I'd build a winback propensity model 
on top of this pipeline. With real-time usage streaming data I'd convert 
this from a batch prediction model to a live early warning system 
triggering retention outreach automatically.

---

## 💼 Business Recommendations

1. **Contract Migration Campaign** — offer Month-to-Month customers incentives to switch to annual plans, targeting tenure < 12 months and MonthlyCharges > median
2. **Billing Stress Early Warning System** — flag customers with PaymentDelinquency + no AutoPay + no Paperless for immediate retention outreach
3. **Support Intensity Monitoring** — customers with high tickets_per_month in first 6 months are at extreme risk; route to dedicated retention agents

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Khushi-Dhargawe/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Generate the dataset (if CSV not present)
python data_generator.py

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Customer_Churn.ipynb
```

---

## 👩‍💻 Author

**Khushi Dhargawe**  
MSc Business Analytics — University College Cork (UCC)  
BE Artificial Intelligence & Machine Learning (Hons. Cybersecurity) — Mumbai University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/khushi-dhargawe)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/Khushi-Dhargawe)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.