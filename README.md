# Customer Churn & Retention Intelligence

An end-to-end data analytics and machine learning project that analyzes customer churn, identifies high-risk customers, estimates revenue at risk, and recommends retention actions.

## Project Overview

Customer churn is a major business problem for subscription-based companies. This project analyzes customer behavior and historical churn patterns to identify customers who are more likely to leave.

The project combines:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PostgreSQL
- SQL
- Power BI
- Machine Learning

The final solution provides business insights, customer risk segmentation, churn prediction, revenue-at-risk analysis, and recommended retention actions.

---

## Business Problem

The objective is to answer the following questions:

1. What percentage of customers are churning?
2. Which customer segments have higher churn?
3. Which factors are associated with customer churn?
4. Which customers are at higher risk of churn?
5. How much monthly revenue is associated with churn?
6. Which customers should receive retention attention first?
7. What retention action can be recommended for high-risk customers?

---

## Dataset

The project uses a Telco Customer Churn dataset.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Total Customers | 7,032 |
| Churned Customers | 1,869 |
| Retained Customers | 5,163 |
| Overall Churn Rate | 26.58% |
| Total Monthly Revenue | $455,661.00 |
| Monthly Revenue Lost | $139,130.85 |
| Annualized Revenue Lost | $1,669,570.20 |

---

## Project Structure

```text
Customer-Churn-Retention-Intelligence
│
├── data
│   ├── raw
│   │   └── Telco_customer_churn.xlsx
│   │
│   └── processed
│       ├── customer_churn_cleaned.csv
│       └── customer_churn_predictions.csv
│
├── python
│   ├── 01_data_inspection.py
│   ├── 02_data_cleaning.py
│   ├── 03_data_quality.py
│   ├── 04_eda.py
│   ├── 05_risk_segmentation.py
│   └── 06_churn_prediction.py
│
├── sql
│   └── 01_churn_analysis.sql
│
├── powerbi
│   └── Customer_Churn_Retention_Intelligence.pbix
│
├── screenshots
│
└── README.md