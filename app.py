import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Scoring System")
st.markdown("Enter customer details to predict creditworthiness")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job Type", 
        options=[0, 1, 2, 3],
        format_func=lambda x: ["Unskilled non-resident", "Unskilled resident", "Skilled", "Highly skilled"][x]
    )
    saving_accounts = st.selectbox("Saving Accounts",
        options=[0, 1, 2, 3],
        format_func=lambda x: ["Little", "Moderate", "Quite Rich", "Rich"][x]
    )
    credit_amount = st.number_input("Credit Amount (€)", min_value=100, max_value=100000, value=5000)
    purpose = st.selectbox("Purpose",
        options=[0, 1, 2, 3, 4, 5, 6, 7],
        format_func=lambda x: ["Car", "Furniture", "Radio/TV", "Domestic", "Repairs", "Education", "Business", "Vacation"][x]
    )

with col2:
    sex = st.selectbox("Gender", options=[0, 1], format_func=lambda x: ["Female", "Male"][x])
    housing = st.selectbox("Housing",
        options=[0, 1, 2],
        format_func=lambda x: ["Free", "Own", "Rent"][x]
    )
    checking_account = st.selectbox("Checking Account",
        options=[0, 1, 2],
        format_func=lambda x: ["Little", "Moderate", "Rich"][x]
    )
    duration = st.number_input("Loan Duration (months)", min_value=1, max_value=120, value=24)

st.divider()

# Predict button
if st.button("🔍 Check Creditworthiness", use_container_width=True):
    # Send to FastAPI
    payload = {
        "age": age,
        "sex": sex,
        "job": job,
        "housing": housing,
        "saving_accounts": saving_accounts,
        "checking_account": checking_account,
        "credit_amount": credit_amount,
        "duration": duration,
        "purpose": purpose
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        st.divider()
        if "APPROVED" in result["credit_decision"]:
            st.success(f"## {result['credit_decision']}")
        else:
            st.error(f"## {result['credit_decision']}")

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Risk Score", f"{result['risk_score'] * 100:.1f}%")
        with col4:
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

        st.info(f"📊 {result['message']}")

    except:
        st.error("❌ API not running! Start the FastAPI server first.")
        st.code("uvicorn src.serve:app --reload")