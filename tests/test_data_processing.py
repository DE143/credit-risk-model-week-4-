from src.data_processing import create_aggregate_features
import pandas as pd

def test_aggregate_features():
    df = pd.DataFrame({
        "CustomerId": [1,1,2],
        "TransactionId":[1,2,3],
        "Amount":[100,200,300]
    })
    agg = create_aggregate_features(df)
    assert "total_amount" in agg.columns
