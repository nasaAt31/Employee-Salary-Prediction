import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("random_forest.pkl")

gender_encoder = joblib.load("gender_encoder.pkl")
education_encoder = joblib.load("education_encoder.pkl")
job_encoder = joblib.load("job_encoder.pkl")

# Page settings
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Prediction")
st.write(
    "Predict employee salary using Machine Learning."
)

st.divider()

# User Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0
)

gender = st.selectbox(
    "Gender",
    list(gender_encoder.classes_)
)

education = st.selectbox(
    "Education Level",
    list(education_encoder.classes_)
)

job_title = st.selectbox(
    "Job Title",
    sorted(job_encoder.classes_)
)

# Prediction
if st.button("Predict Salary"):

    gender_encoded = gender_encoder.transform([gender])[0]
    education_encoded = education_encoder.transform([education])[0]
    job_encoded = job_encoder.transform([job_title])[0]

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "Education Level": [education_encoded],
        "Job Title": [job_encoded],
        "Years of Experience": [experience]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"💰 Predicted Salary: ${prediction:,.2f}"
    )
    