import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_excel(
    "data/raw/Telco_customer_churn.xlsx"
)

print("Original shape:", df.shape)


# ==========================================
# 2. STANDARDIZE COLUMN NAMES
# ==========================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())


# ==========================================
# 3. CONVERT TOTAL CHARGES TO NUMERIC
# ==========================================

df["total_charges"] = pd.to_numeric(
    df["total_charges"],
    errors="coerce"
)

print("\nMissing total_charges:")
print(df["total_charges"].isnull().sum())


# ==========================================
# 4. CHECK DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)


# ==========================================
# 5. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()


# ==========================================
# 6. HANDLE MISSING TOTAL CHARGES
# ==========================================

# These records cannot contribute reliable
# total-charge based analysis.
df = df.dropna(
    subset=["total_charges"]
)


# ==========================================
# 7. CONVERT CHURN TO BINARY
# ==========================================

df["churn"] = (
    df["churn_label"]
    .map({
        "Yes": 1,
        "No": 0
    })
)


# ==========================================
# 8. CHECK CHURN VALUES
# ==========================================

print("\nChurn distribution:")
print(df["churn"].value_counts())


# ==========================================
# 9. CREATE TENURE GROUP
# ==========================================

def create_tenure_group(months):

    if months <= 12:
        return "0-12 Months"

    elif months <= 24:
        return "13-24 Months"

    elif months <= 48:
        return "25-48 Months"

    else:
        return "49+ Months"


df["tenure_group"] = df["tenure_months"].apply(
    create_tenure_group
)


# ==========================================
# 10. CREATE SERVICE COUNT
# ==========================================

service_columns = [
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies"
]

df["service_count"] = (
    df[service_columns]
    .eq("Yes")
    .sum(axis=1)
)


# ==========================================
# 11. CUSTOMER VALUE SEGMENT
# ==========================================

df["customer_value_segment"] = pd.qcut(
    df["total_charges"],
    q=4,
    labels=[
        "Low",
        "Medium",
        "High",
        "VIP"
    ],
    duplicates="drop"
)


# ==========================================
# 12. ESTIMATED ANNUAL REVENUE
# ==========================================

df["annualized_revenue"] = (
    df["monthly_charges"] * 12
)


# ==========================================
# 13. REVENUE CONTRIBUTION
# ==========================================

df["revenue_status"] = np.where(
    df["churn"] == 1,
    "Lost Customer",
    "Active Customer"
)


# ==========================================
# 14. SAVE CLEAN DATA
# ==========================================

output_path = (
    "data/processed/"
    "customer_churn_cleaned.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# 15. FINAL REPORT
# ==========================================

print("\n===================================")
print("DATA CLEANING COMPLETED")
print("===================================")

print("Final shape:", df.shape)

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicates:",
    df.duplicated().sum()
)

print(
    "Saved to:",
    output_path
)