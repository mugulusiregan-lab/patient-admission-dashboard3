import streamlit as st
import pandas as pd
import joblib

st.title("Patient Admission Prediction Dashboard")

model = joblib.load("patient_admission_model.pkl")

st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 0, 100, 30)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

department = st.sidebar.selectbox(
    "Department",
    ["Emergency", "Maternity", "Paediatrics",
     "General", "Gynaecology"]
)

arrival = st.sidebar.selectbox(
    "Arrival Mode",
    ["Walk-in", "Referral", "Ambulance"]
)

triage = st.sidebar.selectbox(
    "Triage Level",
    ["Green", "Yellow", "Red"]
)

duration = st.sidebar.number_input(
    "Symptom Duration (Days)", 0, 365, 2
)

temperature = st.sidebar.number_input(
    "Temperature (°C)", 34.0, 42.0, 37.0
)

pulse = st.sidebar.number_input(
    "Pulse (bpm)", 30, 250, 80
)

spo2 = st.sidebar.number_input(
    "SpO2 (%)", 50, 100, 98
)

bp = st.sidebar.number_input(
    "Systolic BP", 50, 250, 120
)

comorbidity = st.sidebar.selectbox(
    "Known Comorbidity", ["No", "Yes"]
)

if st.button("Predict Admission"):

    patient = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Department": [department],
        "Arrival_Mode": [arrival],
        "Triage_Level": [triage],
        "Symptom_Duration_Days": [duration],
        "Temperature_C": [temperature],
        "Pulse_bpm": [pulse],
        "SpO2_percent": [spo2],
        "Systolic_BP": [bp],
        "Known_Comorbidity": [comorbidity]
    })

    pred = model.predict(patient)[0]

    prob = model.predict_proba(patient)[0][1]

    st.metric(
        "Model Admission Probability",
        f"{prob*100:.1f}%"
    )

    st.write("Predicted simulated label:", pred)

    st.warning(
        "Educational model only. "
        "This is not a clinical admission decision."
    )
    st.balloons()

    