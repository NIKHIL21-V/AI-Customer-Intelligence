# ============================================================
# AI CUSTOMER INTELLIGENCE & PREDICTIVE ANALYTICS PLATFORM
# Customer Churn Prediction Module
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


# ------------------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "cleaned_sales.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "churn_model.joblib"
PREDICTIONS_PATH = REPORT_DIR / "customer_churn_predictions.csv"
METRICS_PATH = REPORT_DIR / "churn_model_metrics.txt"

# A customer is considered inactive if they make no purchase
# during the final 90 days of the observation period.
CHURN_DAYS = 90

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# ------------------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Make sure data/cleaned_sales.csv exists."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
        ]
    ).copy()

    # Use completed purchase transactions only.
    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (
            ~df["InvoiceNo"]
            .astype(str)
            .str.startswith("C")
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid purchase transactions remain."
        )

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    return df


# ------------------------------------------------------------
# 3. CREATE CUSTOMER FEATURES AND CHURN LABELS
# ------------------------------------------------------------

def prepare_customer_data(df):
    observation_end = df["InvoiceDate"].max()

    # Keep the full observation period available in this dataset.
    # The churn label is based on inactivity relative to its end.
    cutoff_date = (
        observation_end.normalize()
        - pd.Timedelta(days=CHURN_DAYS)
    )

    # Customers must have been observed purchasing before the
    # cutoff date to be eligible for the training dataset.
    historical = df[df["InvoiceDate"] < cutoff_date].copy()

    # Purchases in the final 90 days determine the label.
    outcome_period = df[
        df["InvoiceDate"] >= cutoff_date
    ].copy()

    if historical.empty:
        raise ValueError(
            "Not enough historical data to create features."
        )

    features = (
        historical.groupby("CustomerID")
        .agg(
            Last_Purchase=("InvoiceDate", "max"),
            First_Purchase=("InvoiceDate", "min"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Product_Diversity=("StockCode", "nunique")
            if "StockCode" in historical.columns
            else ("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    # Recency is measured at the end of the historical period,
    # not at the end of the outcome period.
    features["Recency"] = (
        cutoff_date - features["Last_Purchase"]
    ).dt.days.clip(lower=0)

    features["Customer_Tenure"] = (
        features["Last_Purchase"]
        - features["First_Purchase"]
    ).dt.days.clip(lower=0)

    features["Average_Order_Value"] = (
        features["Monetary"]
        / features["Frequency"].replace(0, np.nan)
    )

    # Label: 1 means no purchase during the final 90 days;
    # 0 means at least one purchase during those 90 days.
    active_customer_ids = set(
        outcome_period["CustomerID"].unique()
    )

    features["Churn"] = (
        ~features["CustomerID"].isin(active_customer_ids)
    ).astype(int)

    # Remove dates after constructing the numeric features.
    features = features.drop(
        columns=["Last_Purchase", "First_Purchase"]
    )

    return features, cutoff_date, observation_end


# ------------------------------------------------------------
# 4. TRAIN AND EVALUATE THE MODEL
# ------------------------------------------------------------

def train_model(customer_data):
    feature_columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "Total_Quantity",
        "Product_Diversity",
        "Customer_Tenure",
        "Average_Order_Value",
    ]

    X = customer_data[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    y = customer_data["Churn"]

    if y.nunique() < 2:
        raise ValueError(
            "The churn labels contain only one class. "
            "A classifier needs both churned and active customers."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "There are too few examples in one churn class "
            "for a reliable train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    auc = (
        roc_auc_score(y_test, probabilities)
        if y_test.nunique() == 2
        else float("nan")
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, predictions)

    metrics_text = f"""
CUSTOMER CHURN MODEL EVALUATION
================================

Churn definition:
No purchase during the final {CHURN_DAYS} days of the dataset.

Eligible customers: {len(customer_data):,}
Training customers: {len(X_train):,}
Testing customers: {len(X_test):,}

Churn rate: {y.mean():.2%}
Accuracy: {accuracy:.4f}
ROC-AUC: {auc:.4f}

Confusion Matrix:
{matrix}

Classification Report:
{report}

IMPORTANT LIMITATION:
This is a demonstration model trained on historical purchasing
features with an inactivity-based label. A random holdout split
does not establish reliable future-period performance. Use a
strictly time-based validation design before using predictions
for real business decisions.
"""

    return model, feature_columns, metrics_text


# ------------------------------------------------------------
# 5. PREDICT CUSTOMER CHURN RISK
# ------------------------------------------------------------

def create_predictions(model, customer_data, feature_columns):
    result = customer_data.copy()

    X = result[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    result["Churn_Probability"] = model.predict_proba(X)[:, 1]

    result["Churn_Risk"] = pd.cut(
        result["Churn_Probability"],
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    result["Actual_Churn_Label"] = result["Churn"].map(
        {0: "Active", 1: "Inactive"}
    )

    result = result.sort_values(
        "Churn_Probability",
        ascending=False,
    )

    return result


# ------------------------------------------------------------
# 6. MAIN PROGRAM
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("AI CUSTOMER INTELLIGENCE PLATFORM")
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    print("\nStep 1: Loading sales data...")
    sales = load_data()
    print(f"Loaded {len(sales):,} purchase lines.")

    print("\nStep 2: Preparing customer features...")
    customer_data, cutoff_date, observation_end = (
        prepare_customer_data(sales)
    )

    print(f"Eligible customers: {len(customer_data):,}")
    print(f"Historical cutoff: {cutoff_date.date()}")
    print(f"Dataset end: {observation_end.date()}")
    print(
        "Inactive customers:",
        f"{customer_data['Churn'].sum():,}",
    )

    print("\nStep 3: Training machine learning model...")
    model, feature_columns, metrics_text = train_model(
        customer_data
    )

    print("\nStep 4: Evaluating model...")
    print(metrics_text)

    print("\nStep 5: Generating customer risk scores...")
    predictions = create_predictions(
        model,
        customer_data,
        feature_columns,
    )

    # Save the trained pipeline and its feature schema.
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "churn_days": CHURN_DAYS,
            "label_definition": (
                f"No purchase in the final {CHURN_DAYS} days "
                "of the dataset"
            ),
        },
        MODEL_PATH,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    METRICS_PATH.write_text(
        metrics_text,
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Evaluation report saved to: {METRICS_PATH}")

    print("\nTop 10 customers by estimated churn probability:")

    print(
        predictions[
            [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "Churn_Probability",
                "Churn_Risk",
                "Actual_Churn_Label",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()