import pandas as pd
import mlflow.sklearn
import sys

MODEL_URI = "models:/credit-risk/Production"

def predict(input_csv, output_csv):
    model = mlflow.sklearn.load_model(MODEL_URI)
    df = pd.read_csv(input_csv)

    probs = model.predict_proba(df)[:, 1]
    df["risk_probability"] = probs

    df.to_csv(output_csv, index=False)
    print("Predictions saved to", output_csv)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/predict.py input.csv output.csv")
    else:
        predict(sys.argv[1], sys.argv[2])
