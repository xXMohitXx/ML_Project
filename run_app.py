import streamlit as st
import pandas as pd
from joblib import load

# Load the saved model
model = load('lung_cancer_survival_model.joblib')

# Title of the app
st.title("Lung Cancer Survival Prediction")

# Input fields for user data
st.header("Enter Patient Details:")
age = st.number_input("Age", min_value=0, step=1)
gender = st.selectbox("Gender", options=['Male', 'Female'])
country = st.text_input("Country")
diagnosis_date = st.date_input("Diagnosis Date")
cancer_stage = st.selectbox("Cancer Stage", options=['Stage I', 'Stage II', 'Stage III', 'Stage IV'])
beginning_of_treatment_date = st.date_input("Beginning of Treatment Date")
family_history = st.selectbox("Family History of Cancer", options=['Yes', 'No'])
smoking_status = st.selectbox("Smoking Status", options=['Current Smoker', 'Former Smoker', 'Never Smoked', 'Passive Smoker'])
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
cholesterol_level = st.number_input("Cholesterol Level", min_value=0.0, step=0.1)
hypertension = st.selectbox("Hypertension", options=['Yes', 'No'])
asthma = st.selectbox("Asthma", options=['Yes', 'No'])
cirrhosis = st.selectbox("Cirrhosis", options=['Yes', 'No'])
other_cancer = st.selectbox("Other Cancer Diagnosis", options=['Yes', 'No'])
treatment_type = st.selectbox("Treatment Type", options=['Chemotherapy', 'Combined', 'Radiation', 'Surgery'])
end_treatment_date = st.date_input("End Treatment Date")

# Preprocess user inputs
gender = 1 if gender == 'Male' else 0
family_history = 1 if family_history == 'Yes' else 0
hypertension = 1 if hypertension == 'Yes' else 0
asthma = 1 if asthma == 'Yes' else 0
cirrhosis = 1 if cirrhosis == 'Yes' else 0
other_cancer = 1 if other_cancer == 'Yes' else 0
cancer_stage = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4}[cancer_stage]
smoking_status = {'Current Smoker': 1, 'Former Smoker': 2, 'Never Smoked': 3, 'Passive Smoker': 4}[smoking_status]
treatment_type = {'Chemotherapy': 1, 'Combined': 2, 'Radiation': 3, 'Surgery': 4}[treatment_type]

# Calculate days under treatment
days_under_treatment = (pd.to_datetime(end_treatment_date) - pd.to_datetime(beginning_of_treatment_date)).days

# Prediction button
if st.button("Predict"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'cancer_stage': [cancer_stage],
        'family_history': [family_history],
        'smoking_status': [smoking_status],
        'bmi': [bmi],
        'cholesterol_level': [cholesterol_level],
        'hypertension': [hypertension],
        'asthma': [asthma],
        'cirrhosis': [cirrhosis],
        'other_cancer': [other_cancer],
        'treatment_type': [treatment_type],
        'days_under_treatment': [days_under_treatment]
    })

    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display results
    if prediction == 1:
        st.markdown(f"### **Prediction: The patient survived.**")
    else:
        st.markdown(f"### **Prediction: The patient did not survive.**")

    st.markdown(f"### Survival Probability: **{probability * 100:.2f}%**")
