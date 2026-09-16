import pandas as pd

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/customer_churn_cleaned.csv"
)

print("\n===================================")
print("CLEANED DATASET CHECK")
print("===================================")

print("\nShape:")
print(df.shape)

print("\nMissing values by column:")
missing = df.isnull().sum()

print(
    missing[missing > 0]
    .sort_values(ascending=False)
)

print("\nMissing percentage by column:")
missing_pct = (
    df.isnull().mean() * 100
).round(2)

print(
    missing_pct[missing_pct > 0]
    .sort_values(ascending=False)
)

print("\nChurn distribution:")
print(
    df["churn"].value_counts()
)

print("\nChurn rate:")
print(
    round(df["churn"].mean() * 100, 2),
    "%"
)

print("\nTenure groups:")
print(
    df["tenure_group"].value_counts()
)

print("\nCustomer value segments:")
print(
    df["customer_value_segment"].value_counts()
)

print("\nService count:")
print(
    df["service_count"].describe()
)

print("\nRevenue status:")
print(
    df["revenue_status"].value_counts()
)