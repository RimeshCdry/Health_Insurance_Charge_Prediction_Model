import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model and scaler with error handling
def load_model_and_scaler():
    try:
        model = joblib.load("insurance_prediction_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# Streamlit UI
st.set_page_config(page_title="Insurance Prediction App 🏥💰", page_icon="💰")
st.title('Insurance Prediction App 🏥💰')
st.write('Predict insurance charges based on input features')

# User Input Fields
age = st.number_input('Age', min_value=18, max_value=150, value=25)
sex = st.radio('Sex', ["Male", "Female"], horizontal=True)
bmi = st.number_input('BMI', min_value=10.0, max_value=100.0, value=25.0, step=0.1)
children = st.slider('Children', min_value=0, max_value=14, value=0)
smoker = st.radio('Smoker', ['Yes', 'No'], horizontal=True)
region = st.selectbox('Region', ['Southwest', 'Southeast', 'Northwest', 'Northeast'])

# Convert categorical inputs to numerical
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_mapping = { 'Northeast': 0, 'Northwest': 1,'Southeast': 2,'Southwest': 3} 
region = region_mapping[region]

# Prepare input data
feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=feature_names)

# Standardize only 'age' and 'bmi'
columns_to_scale = ['age', 'bmi']
try:
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])
except Exception as e:
    st.error(f"Error in data scaling: {e}")
    st.stop()

# Prediction Button
if st.button('Predict 💡'):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f'💰 Predicted Insurance Charge: **${prediction:,.2f}**')
    except Exception as e:
        st.error(f"Error in prediction: {e}")
