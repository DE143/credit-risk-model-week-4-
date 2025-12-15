import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["TransactionStartTime"])


# ---------------------------------------------------
# 2. CREATE RFM FEATURES
# ---------------------------------------------------
def calculate_rfm(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    snapshot_date = pd.to_datetime(snapshot_date)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    return rfm


# ---------------------------------------------------
# 3. CLUSTER CUSTOMERS (K-MEANS)
# ---------------------------------------------------
def cluster_rfm(rfm: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm


# ---------------------------------------------------
# 4. IDENTIFY HIGH-RISK CLUSTER
# ---------------------------------------------------
def assign_high_risk_label(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    High-risk cluster is the one with:
    - Lowest Frequency
    - Lowest Monetary value
    """

    cluster_summary = (
        rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .sort_values(["Frequency", "Monetary"], ascending=[True, True])
    )

    high_risk_cluster = cluster_summary.index[0]

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]


# ---------------------------------------------------
# 5. MERGE TARGET BACK TO DATASET
# ---------------------------------------------------
def merge_target(df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(target_df, on="CustomerId", how="left")
    return df


# ---------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------
def run_proxy_target_engineering(
    raw_path="data/raw/data.csv",
    output_path="data/processed/transactions_with_target.csv",
):
    df = load_data(raw_path)

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = calculate_rfm(df, snapshot_date)
    rfm = cluster_rfm(rfm)
    target_df = assign_high_risk_label(rfm)

    final_df = merge_target(df, target_df)

    final_df.to_csv(output_path, index=False)

    print("✅ Proxy target variable created")
    print("Target column: is_high_risk")
    print("Saved to:", output_path)


# ---------------------------------------------------
# 7. SCRIPT ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    run_proxy_target_engineering()
