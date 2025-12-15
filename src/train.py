import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ---------------------------------------------------
# 1. LOAD DATA & PREPROCESSING
# ---------------------------------------------------
df = pd.read_csv("data/processed/transactions_with_target.csv")

# Drop ID columns not used for modeling
drop_cols = [
    "TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId", "TransactionStartTime"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Separate features and target
X = df.drop("is_high_risk", axis=1)
y = df["is_high_risk"]

# *** FIX 2: Convert Target Variable to Integer (This resolves the XGBoost ValueError) ***
# If 'is_high_risk' is read as float (0.0, 1.0), XGBoost defaults to regression.
y = y.astype(int) 

# Convert categorical columns to string (safer for get_dummies)
for col in X.select_dtypes("object").columns:
    X[col] = X[col].astype(str)

# *** FIX 1: One-Hot Encode Categorical Features ***
X = pd.get_dummies(X, drop_first=True) 

print(f"Features after One-Hot Encoding: {X.shape[1]}")

# ---------------------------------------------------
# 2. TRAIN/TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------
# 3. MLflow EXPERIMENT SETUP
# ---------------------------------------------------
mlflow.set_experiment("credit-risk-model")

# ---------------------------------------------------
# 4. TRAINING FUNCTION
# ---------------------------------------------------
def train_and_log_model(model, model_name, param_grid=None):
    with mlflow.start_run(run_name=model_name):
        # Hyperparameter tuning if provided
        if param_grid:
            # GridSearchCV uses 'roc_auc' scoring, requiring predict_proba
            grid = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc")
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            
            # Log all parameters, including the ones tuned
            mlflow.log_params(grid.best_params_)
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        # Predict & Evaluate
        y_pred = best_model.predict(X_test)
        # Use predict_proba for ROC-AUC score
        y_prob = best_model.predict_proba(X_test)[:, 1] 

        # Calculate metrics (zero_division=0 handles cases where no samples are predicted for a class)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"✅ {model_name} trained and logged")
        print(f"ROC-AUC: {roc_auc:.4f}")

        return best_model

# ---------------------------------------------------
# 5. TRAIN LOGISTIC REGRESSION
# ---------------------------------------------------
lr_model = LogisticRegression(max_iter=5000, random_state=42)
lr_param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}
best_lr = train_and_log_model(lr_model, "LogisticRegression", lr_param_grid)

# ---------------------------------------------------
# 6. TRAIN XGBOOST
# ---------------------------------------------------
xgb_model = XGBClassifier(
    objective='binary:logistic', # Explicitly set for binary classification
    use_label_encoder=False, 
    eval_metric="logloss", 
    random_state=42
)
xgb_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}
best_xgb = train_and_log_model(xgb_model, "XGBoost", xgb_param_grid)