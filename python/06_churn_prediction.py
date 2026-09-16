import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/customer_churn_cleaned.csv"
)

print("Dataset shape:", df.shape)


# ==========================================
# SELECT FEATURES
# ==========================================

features = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "service_count"
]

X = df[features]

y = df["churn"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())



# ==========================================
# FEATURE TYPES
# ==========================================

categorical_features = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method"
]

numerical_features = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "service_count"
]




# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)



# ==========================================
# LOGISTIC REGRESSION
# ==========================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = (
    logistic_model.predict(X_test)
)

logistic_probabilities = (
    logistic_model.predict_proba(X_test)[:, 1]
)



# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(
    name,
    y_true,
    predictions,
    probabilities
):

    print("\n===================================")
    print(name)
    print("===================================")

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions
    )

    recall = recall_score(
        y_true,
        predictions
    )

    f1 = f1_score(
        y_true,
        predictions
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    logistic_predictions,
    logistic_probabilities
)


# ==========================================
# RANDOM FOREST
# ==========================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

rf_predictions = (
    random_forest_model.predict(X_test)
)

rf_probabilities = (
    random_forest_model
    .predict_proba(X_test)[:, 1]
)

evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_predictions,
    rf_probabilities
)



# ==========================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

print("\n===================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("===================================")

# Get the fitted preprocessing pipeline
fitted_preprocessor = (
    random_forest_model
    .named_steps["preprocessor"]
)

# Get transformed feature names
feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

# Get Random Forest model
rf_model = (
    random_forest_model
    .named_steps["model"]
)

# Get feature importance
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print(
    feature_importance.head(20)
)


import matplotlib.pyplot as plt

# ==========================================
# TOP 15 FEATURE IMPORTANCE
# ==========================================

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        "importance",
        ascending=True
    )
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title(
    "Top 15 Features Influencing Churn Prediction"
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.show()



# ==========================================
# CUSTOMER-LEVEL CHURN PREDICTIONS
# ==========================================

print("\n===================================")
print("CUSTOMER-LEVEL CHURN PREDICTIONS")
print("===================================")

# Predict probability for every customer
all_probabilities = (
    random_forest_model
    .predict_proba(X)[:, 1]
)

# Predict churn class
all_predictions = (
    random_forest_model
    .predict(X)
)

prediction_df = df[
    [
        "customerid",
        "tenure_months",
        "contract",
        "internet_service",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "tech_support",
        "online_security"
    ]
].copy()

prediction_df["churn_probability"] = (
    all_probabilities * 100
).round(2)

prediction_df["predicted_churn"] = (
    all_predictions
)

# ==========================================
# ML RISK CATEGORY
# ==========================================

def ml_risk_category(probability):

    if probability >= 75:
        return "Critical"

    elif probability >= 50:
        return "High"

    elif probability >= 25:
        return "Medium"

    else:
        return "Low"


prediction_df["ml_risk_category"] = (
    prediction_df["churn_probability"]
    .apply(ml_risk_category)
)


# ==========================================
# EXPECTED REVENUE AT RISK
# ==========================================

prediction_df["expected_monthly_revenue_at_risk"] = (
    prediction_df["churn_probability"] / 100
    * prediction_df["monthly_charges"]
)

prediction_df["expected_monthly_revenue_at_risk"] = (
    prediction_df["expected_monthly_revenue_at_risk"]
    .round(2)
)


# ==========================================
# RETENTION PRIORITY
# ==========================================

def retention_priority(row):

    if (
        row["ml_risk_category"] == "Critical"
        and row["expected_monthly_revenue_at_risk"] >= 75
    ):
        return "P1 - Immediate"

    elif (
        row["ml_risk_category"] in ["Critical", "High"]
        and row["expected_monthly_revenue_at_risk"] >= 50
    ):
        return "P2 - High"

    elif row["ml_risk_category"] in ["Critical", "High"]:
        return "P3 - Medium"

    else:
        return "P4 - Monitor"


prediction_df["retention_priority"] = (
    prediction_df.apply(
        retention_priority,
        axis=1
    )
)


# ==========================================
# RETENTION RECOMMENDATION
# ==========================================

def retention_action(row):

    if row["contract"] == "Month-to-month":
        return "Offer annual contract incentive"

    elif row["tech_support"] == "No":
        return "Offer Tech Support trial/bundle"

    elif row["online_security"] == "No":
        return "Offer Online Security bundle"

    elif row["payment_method"] == "Electronic check":
        return "Promote automatic payment"

    elif row["internet_service"] == "Fiber optic":
        return "Review Fiber pricing/service quality"

    else:
        return "Standard retention campaign"


prediction_df["retention_action"] = (
    prediction_df.apply(
        retention_action,
        axis=1
    )
)


# ==========================================
# RETENTION PRIORITY SUMMARY
# ==========================================

print("\n--- RETENTION PRIORITY SUMMARY ---")

priority_summary = (
    prediction_df[
        prediction_df["retention_priority"] != "P4 - Monitor"
    ]
    .groupby("retention_priority")
    .agg(
        customers=("customerid", "count"),
        avg_churn_probability=("churn_probability", "mean"),
        revenue_at_risk=(
            "expected_monthly_revenue_at_risk",
            "sum"
        )
    )
    .reset_index()
)

priority_summary["avg_churn_probability"] = (
    priority_summary["avg_churn_probability"]
    .round(2)
)

priority_summary["revenue_at_risk"] = (
    priority_summary["revenue_at_risk"]
    .round(2)
)

print(priority_summary)


# ==========================================
# SAVE PREDICTIONS
# ==========================================

output_path = (
    "data/processed/"
    "customer_churn_predictions.csv"
)

prediction_df.to_csv(
    output_path,
    index=False
)

print(
    "\nSaved predictions to:",
    output_path
)


# ==========================================
# TOP 20 RETENTION PRIORITIES
# ==========================================

top_risk = (
    prediction_df
    .sort_values(
        "expected_monthly_revenue_at_risk",
        ascending=False
    )
    .head(20)
)

print("\n--- TOP 20 RETENTION PRIORITIES ---")

print(
    top_risk[
        [
            "customerid",
            "churn_probability",
            "ml_risk_category",
            "expected_monthly_revenue_at_risk",
            "retention_priority",
            "contract",
            "tenure_months",
            "internet_service",
            "payment_method",
            "retention_action"
        ]
    ].to_string(index=False)
)