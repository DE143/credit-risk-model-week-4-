from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from .pydantic_models import CreditRequest, CreditResponse

app = FastAPI(title="Credit Risk Prediction API")

# Load best model from MLflow Model Registry
MODEL_URI = "models:/credit-risk/Production"
model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running"}

@app.post("/predict", response_model=CreditResponse)
def predict(data: CreditRequest):
    # Convert request to DataFrame
    df = pd.DataFrame([data.dict()])
    # Predict probability of high risk
    prob = model.predict_proba(df)[0][1]
    return {"risk_probability": prob}
