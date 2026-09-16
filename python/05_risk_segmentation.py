import pandas as pd

# ==========================================
# LOAD CLEANED DATA
# ==========================================

df = pd.read_csv(
    "data/processed/customer_churn_cleaned.csv"
)

print("Dataset shape:", df.shape)


# ==========================================
# HIGH-RISK SEGMENT
# ==========================================

df["high_risk_segment"] = (
    (df["contract"] == "Month-to-month") &
    (df["tenure_months"] <= 12) &
    (df["internet_service"] == "Fiber optic") &
    (df["payment_method"] == "Electronic check")
)

print("\n--- HIGH-RISK SEGMENT ---")

high_risk = df[
    df["high_risk_segment"]
]

print(
    "High-risk customers:",
    len(high_risk)
)

print(
    "Churned customers:",
    high_risk["churn"].sum()
)

print(
    "Churn rate:",
    round(
        high_risk["churn"].mean() * 100,
        2
    ),
    "%"
)

# ==========================================
# REVENUE AT RISK
# ==========================================

high_risk_revenue = (
    high_risk
    .loc[
        high_risk["churn"] == 1,
        "monthly_charges"
    ]
    .sum()
)

print(
    "\nMonthly revenue associated with churned "
    "high-risk customers:",
    round(high_risk_revenue, 2)
)


# ==========================================
# OVERALL VS HIGH-RISK
# ==========================================

overall_churn = df["churn"].mean() * 100

high_risk_churn = (
    high_risk["churn"].mean() * 100
)

print("\n--- RISK COMPARISON ---")

print(
    "Overall churn rate:",
    round(overall_churn, 2),
    "%"
)

print(
    "High-risk segment churn rate:",
    round(high_risk_churn, 2),
    "%"
)

print(
    "Risk multiplier:",
    round(
        high_risk_churn / overall_churn,
        2
    ),
        "x"
)


# ==========================================
# EXPLAINABLE CUSTOMER RISK SCORE
# ==========================================

df["risk_score"] = 0


# ------------------------------------------
# CONTRACT RISK
# ------------------------------------------

df.loc[
    df["contract"] == "Month-to-month",
    "risk_score"
] += 25

df.loc[
    df["contract"] == "One year",
    "risk_score"
] += 10


# ------------------------------------------
# TENURE RISK
# ------------------------------------------

df.loc[
    df["tenure_months"] <= 12,
    "risk_score"
] += 20

df.loc[
    (df["tenure_months"] > 12) &
    (df["tenure_months"] <= 24),
    "risk_score"
] += 12


# ------------------------------------------
# PAYMENT RISK
# ------------------------------------------

df.loc[
    df["payment_method"] == "Electronic check",
    "risk_score"
] += 20


# ------------------------------------------
# INTERNET SERVICE RISK
# ------------------------------------------

df.loc[
    df["internet_service"] == "Fiber optic",
    "risk_score"
] += 15


# ------------------------------------------
# MONTHLY CHARGE RISK
# ------------------------------------------

charge_75 = df["monthly_charges"].quantile(0.75)

df.loc[
    df["monthly_charges"] >= charge_75,
    "risk_score"
] += 10


# ------------------------------------------
# TECH SUPPORT RISK
# ------------------------------------------

df.loc[
    (df["internet_service"] != "No") &
    (df["tech_support"] == "No"),
    "risk_score"
] += 5


# ------------------------------------------
# ONLINE SECURITY RISK
# ------------------------------------------

df.loc[
    (df["internet_service"] != "No") &
    (df["online_security"] == "No"),
    "risk_score"
] += 5



# ==========================================
# RISK CATEGORY
# ==========================================

def risk_category(score):

    if score >= 75:
        return "Critical"

    elif score >= 50:
        return "High"

    elif score >= 25:
        return "Medium"

    else:
        return "Low"


df["risk_category"] = (
    df["risk_score"]
    .apply(risk_category)
)




# ==========================================
# RISK DISTRIBUTION
# ==========================================

print("\n--- RISK CATEGORY DISTRIBUTION ---")

risk_distribution = (
    df.groupby("risk_category")["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
    .reset_index()
)

risk_distribution["churn_rate"] = (
    risk_distribution["churn_rate"] * 100
).round(2)

print(risk_distribution)




print("\n--- RISK SCORE SUMMARY ---")

print(
    df["risk_score"]
    .describe()
    .round(2)
)



# ==========================================
# CRITICAL CUSTOMERS
# ==========================================

critical_customers = df[
    df["risk_category"] == "Critical"
].copy()

print("\n--- CRITICAL CUSTOMERS ---")

print(
    "Critical customers:",
    len(critical_customers)
)

print(
    "Churned critical customers:",
    critical_customers["churn"].sum()
)

print(
    "Critical churn rate:",
    round(
        critical_customers["churn"].mean() * 100,
        2
    ),
    "%"
)



# ==========================================
# REVENUE AT RISK
# ==========================================

revenue_at_risk = (
    df[
        (df["risk_category"].isin(["High", "Critical"])) &
        (df["churn"] == 0)
    ]["monthly_charges"]
    .sum()
)

print(
    "\nMonthly revenue at risk from active "
    "High/Critical customers:",
    round(revenue_at_risk, 2)
)

print(
    "Annualized revenue at risk:",
    round(revenue_at_risk * 12, 2)
)




