import pandas as pd

df = pd.read_excel("data/raw/Telco_customer_churn.xlsx")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

print("\n--- CHURN DISTRIBUTION ---")
print(df["Churn Label"].value_counts())

print("\n--- CHURN PERCENTAGE ---")
print(
    df["Churn Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n--- CHURN VALUE DISTRIBUTION ---")
print(df["Churn Value"].value_counts())

print("\n--- CHURN LABEL vs CHURN VALUE ---")
print(pd.crosstab(df["Churn Label"], df["Churn Value"]))


# Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

print("\n--- TOTAL CHARGES DATA TYPE ---")
print(df["Total Charges"].dtype)

print("\n--- MISSING TOTAL CHARGES ---")
print(df["Total Charges"].isnull().sum())


print("\n--- CONTRACT TYPES ---")
print(df["Contract"].value_counts())


print("\n--- PAYMENT METHODS ---")
print(df["Payment Method"].value_counts())


print("\n--- INTERNET SERVICES ---")
print(df["Internet Service"].value_counts())


print("\n--- TOP CHURN REASONS ---")
print(
    df.loc[
        df["Churn Label"] == "Yes",
        "Churn Reason"
    ].value_counts()
)


print("\n--- CLTV SUMMARY ---")
print(df["CLTV"].describe())


print("\n--- CHURN SCORE SUMMARY ---")
print(df["Churn Score"].describe())