import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load(r"C:\Users\kisho\OneDrive\Desktop\patient\readmission_model.pkl")
scaler = joblib.load(r"C:\Users\kisho\OneDrive\Desktop\patient\scaler.pkl")

# Load feature names used in training
feature_names = joblib.load(r"C:\Users\kisho\OneDrive\Desktop\patient\feature_names.pkl")  # Save this in training script

# Streamlit UI
st.title("Patient Readmission Prediction")
st.write("Enter patient details to predict if they will be readmitted within 30 days.")

# User Input Form
time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=5)
n_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=200, value=40)
n_procedures = st.number_input("Number of Procedures", min_value=0, max_value=10, value=2)
n_medications = st.number_input("Number of Medications", min_value=0, max_value=50, value=10)
n_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, max_value=10, value=0)
n_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, max_value=10, value=1)
n_emergency = st.number_input("Number of Emergency Visits", min_value=0, max_value=10, value=0)

# Create input DataFrame with correct feature names
input_data = pd.DataFrame([[time_in_hospital, n_lab_procedures, n_procedures, n_medications,
                            n_outpatient, n_inpatient, n_emergency]],
                          columns=['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications',
                                   'n_outpatient', 'n_inpatient', 'n_emergency'])

# Ensure input data matches trained model feature names
missing_features = set(feature_names) - set(input_data.columns)
for feature in missing_features:
    input_data[feature] = 0  # Add missing columns with default values

# Standardize input
input_scaled = scaler.transform(input_data[feature_names])  # Use exact feature set

# Predict button
if st.button("Predict Readmission"):
    prediction = model.predict(input_scaled)
    result = "Readmitted within 30 days" if prediction[0] == 1 else "Not Readmitted"
    st.success(f"Prediction: {result}")
