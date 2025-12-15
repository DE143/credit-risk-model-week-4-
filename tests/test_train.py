def test_train_shapes():
    import pandas as pd
    df = pd.DataFrame({
        "Amount": [100,200,300],
        "avg_amount":[100,200,300],
        "transaction_count":[1,1,1],
        "std_amount":[0,0,0],
        "is_high_risk":[0,1,0]
    })
    X = df.drop("is_high_risk", axis=1)
    y = df["is_high_risk"]
    assert X.shape[0] == y.shape[0]
