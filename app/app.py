from pathlib import Path
import pickle
import warnings

import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "TenureGroup",
]


@st.cache_resource
def load_artifacts():
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*serialized model.*",
    )

    with (BASE_DIR / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)

    with (BASE_DIR / "encoders.pkl").open("rb") as encoder_file:
        encoders = pickle.load(encoder_file)

    return model, encoders


def encode_value(encoders, column_name, value):
    if column_name not in encoders:
        return value
    return encoders[column_name].transform([value])[0]


def get_tenure_group(tenure):
    if tenure <= 12:
        return 0
    if tenure <= 24:
        return 1
    if tenure <= 48:
        return 2
    return 3


model, encoders = load_artifacts()

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict churn risk.")

gender = st.selectbox("Gender", list(encoders["gender"].classes_))
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", list(encoders["Partner"].classes_))
dependents = st.selectbox("Dependents", list(encoders["Dependents"].classes_))
tenure = st.slider("Tenure (months)", 1, 72, 12)

phone_service = st.selectbox("Phone Service", list(encoders["PhoneService"].classes_))
multiple_lines_options = (
    ["No phone service"]
    if phone_service == "No"
    else ["No", "Yes"]
)
multiple_lines = st.selectbox("Multiple Lines", multiple_lines_options)

internet_service = st.selectbox(
    "Internet Service",
    list(encoders["InternetService"].classes_),
)
internet_dependent_options = (
    ["No internet service"]
    if internet_service == "No"
    else ["No", "Yes"]
)
online_security = st.selectbox("Online Security", internet_dependent_options)
online_backup = st.selectbox("Online Backup", internet_dependent_options)
device_protection = st.selectbox("Device Protection", internet_dependent_options)
tech_support = st.selectbox("Tech Support", internet_dependent_options)
streaming_tv = st.selectbox("Streaming TV", internet_dependent_options)
streaming_movies = st.selectbox("Streaming Movies", internet_dependent_options)

contract = st.selectbox("Contract Type", list(encoders["Contract"].classes_))
paperless_billing = st.selectbox(
    "Paperless Billing",
    list(encoders["PaperlessBilling"].classes_),
)
payment_method = st.selectbox(
    "Payment Method",
    list(encoders["PaymentMethod"].classes_),
)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0, 0.1)
total_charges = st.number_input(
    "Total Charges",
    0.0,
    10000.0,
    float(round(monthly_charges * tenure, 2)),
    0.1,
)

tenure_group = get_tenure_group(tenure)

raw_input = {
    "gender": gender,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "TenureGroup": tenure_group,
}

encoded_input = {
    column: encode_value(encoders, column, raw_input[column])
    for column in MODEL_FEATURES
}
input_data = pd.DataFrame([encoded_input], columns=MODEL_FEATURES)

if st.button("Predict Churn"):
    try:
        prob = float(model.predict_proba(input_data)[0][1])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        threshold = 0.4
        prediction = 1 if prob > threshold else 0

        st.subheader(f"Churn Probability: {prob:.2f}")
        st.progress(prob)

        if prediction == 1:
            st.error("Customer likely to churn")

            if prob > 0.75:
                st.write("Very High Risk -> Immediate retention action needed")
            elif prob > 0.55:
                st.write("Moderate Risk -> Offer discounts or incentives")
            else:
                st.write("Mild Risk -> Monitor behavior")
        else:
            st.success("Customer likely to stay")
