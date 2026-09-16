import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/customer_churn_cleaned.csv"
)

print("Dataset shape:", df.shape)


print("\n--- CHURN BY CONTRACT ---")

contract_churn = (
    df.groupby("contract")["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
    .reset_index()
)

contract_churn["churn_rate"] = (
    contract_churn["churn_rate"] * 100
).round(2)

print(contract_churn)

# ==========================================
# CHURN BY TENURE GROUP
# ==========================================

print("\n--- CHURN BY TENURE GROUP ---")

tenure_churn = (
    df.groupby("tenure_group", observed=True)["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
    .reset_index()
)

tenure_churn["churn_rate"] = (
    tenure_churn["churn_rate"] * 100
).round(2)

print(tenure_churn)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=tenure_churn,
    x="tenure_group",
    y="churn_rate"
)

plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()



# ==========================================
# CHURN BY PAYMENT METHOD
# ==========================================

print("\n--- CHURN BY PAYMENT METHOD ---")

payment_churn = (
    df.groupby("payment_method")["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
    .reset_index()
)

payment_churn["churn_rate"] = (
    payment_churn["churn_rate"] * 100
).round(2)

print(payment_churn)


plt.figure(figsize=(10, 5))

sns.barplot(
    data=payment_churn,
    x="payment_method",
    y="churn_rate"
)

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()



# ==========================================
# CHURN BY INTERNET SERVICE
# ==========================================

print("\n--- CHURN BY INTERNET SERVICE ---")

internet_churn = (
    df.groupby("internet_service")["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
    .reset_index()
)

internet_churn["churn_rate"] = (
    internet_churn["churn_rate"] * 100
).round(2)

print(internet_churn)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=internet_churn,
    x="internet_service",
    y="churn_rate"
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# ==========================================
# CHURN BY MONTHLY CHARGES
# ==========================================

print("\n--- MONTHLY CHARGES BY CHURN ---")

charges_churn = (
    df.groupby("churn")["monthly_charges"]
    .agg(
        customers="count",
        average="mean",
        median="median",
        minimum="min",
        maximum="max"
    )
    .round(2)
)

print(charges_churn)



plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="churn",
    y="monthly_charges"
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn (0 = Retained, 1 = Churned)")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()


# ==========================================
# CHURN BY SERVICE COUNT
# ==========================================

print("\n--- CHURN BY SERVICE COUNT ---")

service_churn = (
    df.groupby("service_count")["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
    .reset_index()
)

service_churn["churn_rate"] = (
    service_churn["churn_rate"] * 100
).round(2)

print(service_churn)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=service_churn,
    x="service_count",
    y="churn_rate"
)

plt.title("Churn Rate by Number of Services")
plt.xlabel("Number of Services")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# ==========================================
# CHURN BY INDIVIDUAL SERVICES
# ==========================================

service_features = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support"
]

for service in service_features:

    print(f"\n--- CHURN BY {service.upper()} ---")

    service_analysis = (
        df.groupby(service)["churn"]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean"
        )
        .reset_index()
    )

    service_analysis["churn_rate"] = (
        service_analysis["churn_rate"] * 100
    ).round(2)

    print(service_analysis)


    # ==========================================
# CHURN REASONS ANALYSIS
# ==========================================

print("\n--- TOP CHURN REASONS ---")

churn_reasons = (
    df[df["churn"] == 1]
    ["churn_reason"]
    .value_counts()
    .reset_index()
)

churn_reasons.columns = [
    "churn_reason",
    "customers"
]

print(churn_reasons)


# ==========================================
# TOP 10 CHURN REASONS
# ==========================================

top_reasons = churn_reasons.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_reasons,
    x="customers",
    y="churn_reason"
)

plt.title("Top 10 Customer Churn Reasons")
plt.xlabel("Number of Churned Customers")
plt.ylabel("Churn Reason")

plt.tight_layout()
plt.show()