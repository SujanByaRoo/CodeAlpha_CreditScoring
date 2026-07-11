from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import yaml

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="Predicts creditworthiness using ML",
    version="1.0.0"
)

# 🧠 BaseModel = defines exactly what input data looks like
# If someone sends wrong data, FastAPI auto-rejects it with clear error
class CustomerData(BaseModel):
    age: float
    sex: int           # 0=female, 1=male (encoded)
    job: int           # 0-3 skill level
    housing: int       # 0=free, 1=own, 2=rent
    saving_accounts: int     # 0=little, 1=moderate, 2=quite rich, 3=rich
    checking_account: int    # 0=little, 1=moderate, 2=rich
    credit_amount: float
    duration: float
    purpose: int       # 0-7 encoded purpose

# Load model and scaler once when server starts
# 🧠 We load ONCE at startup, not every request — much faster
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.get("/")
def root():
    return {"message": "Credit Scoring API is running 🚀"}

@app.post("/predict")
def predict(customer: CustomerData):
    """
    This is the maiun endpoint.
    Receives customer data → engineers features → scales → predicts
    Returns: credit decision + probability + reason
    """
    # Feature engineering same as preprocess.py
    credit_per_month = customer.credit_amount / customer.duration
    age_duration_ratio = customer.age / customer.duration

    # Build feature array in same order as training
    features = np.array([[
        customer.age,
        customer.sex,
        customer.job,
        customer.housing,
        customer.saving_accounts,
        customer.checking_account,
        customer.credit_amount,
        customer.duration,
        customer.purpose,
        credit_per_month,
        age_duration_ratio
    ]])

    # Scale using same scaler from training
    features_scaled = scaler.transform(features)

    # Get prediction and probability
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    return {
        "credit_decision": "APPROVED ✅" if prediction == 0 else "REJECTED ❌",
        "risk_score": round(float(probability[1]), 4),  # probability of being bad customer
        "confidence": round(float(max(probability)), 4),
        "message": "Low risk customer" if prediction == 0 else "High risk customer"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "random_forest"}