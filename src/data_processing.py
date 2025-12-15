import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------
# 2. AGGREGATE CUSTOMER FEATURES
# ---------------------------------------------------
def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_amount=("Amount", "std"),
        )
        .reset_index()
    )

    agg_df["std_amount"] = agg_df["std_amount"].fillna(0)
    return agg_df


# ---------------------------------------------------
# 3. DATETIME FEATURES
# ---------------------------------------------------
def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["txn_hour"] = df["TransactionStartTime"].dt.hour
    df["txn_day"] = df["TransactionStartTime"].dt.day
    df["txn_month"] = df["TransactionStartTime"].dt.month
    df["txn_year"] = df["TransactionStartTime"].dt.year

    return df


# ---------------------------------------------------
# 4. PREPROCESSING PIPELINE
# ---------------------------------------------------
def build_preprocessor(numerical_features, categorical_features):

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )

    return preprocessor


# ---------------------------------------------------
# 5. MAIN FEATURE ENGINEERING FUNCTION
# ---------------------------------------------------
def run_feature_engineering(
    raw_path="data/raw/data.csv",
    output_path="data/processed/model_data.csv",
):

    # Load
    df = load_data(raw_path)

    # Datetime features
    df = extract_datetime_features(df)

    # Aggregate features
    agg_df = create_aggregate_features(df)

    # Merge back
    df = df.merge(agg_df, on="CustomerId", how="left")

    # Drop IDs not used for modeling
    drop_cols = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "TransactionStartTime",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Feature groups
    numerical_features = [
        "Amount",
        "Value",
        "txn_hour",
        "txn_day",
        "txn_month",
        "txn_year",
        "total_amount",
        "avg_amount",
        "transaction_count",
        "std_amount",
    ]

    categorical_features = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]

    # Preprocessing
    preprocessor = build_preprocessor(
        numerical_features, categorical_features
    )

    X = preprocessor.fit_transform(df)

    # Save as numpy array (model-ready)
    np.save(output_path.replace(".csv", ".npy"), X)

    print("✅ Feature engineering completed")
    print(f"Saved processed data to {output_path.replace('.csv','.npy')}")


# ---------------------------------------------------
# 6. SCRIPT ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    run_feature_engineering()
