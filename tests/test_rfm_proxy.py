from src.rfm_proxy import calculate_rfm
import pandas as pd

def test_calculate_rfm():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": [1, 2, 3],
        "Amount": [100, 200, 300],
        "TransactionStartTime": pd.to_datetime(
            ["2024-01-01", "2024-01-10", "2024-01-05"]
        )
    })

    rfm = calculate_rfm(df, snapshot_date="2024-01-11")

    assert rfm.shape[0] == 2
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
