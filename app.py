import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
st.title("Rainfall Prediction in India")
st.write("Enter the monthly rainfall data and select the division to predict the annual rainfall.")
model_path = "rainfall_prediction_model.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure the model is present in the app directory.")
    st.stop()
with open(model_path, "rb") as file:
    loaded_model, feature_names = pickle.load(file)
monthly_rainfall = {}
for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
    monthly_rainfall[month] = st.number_input(f"{month} rainfall (in mm)", min_value=0.0, max_value=1000.0, value=0.0)
divisions = ['ANDAMAN & NICOBAR ISLANDS', 'ARUNACHAL PRADESH', 'ASSAM & MEGHALAYA', 'BIHAR', 'CHHATTISGARH',
             'COASTAL ANDHRA PRADESH', 'COASTAL KARNATAKA', 'EAST MADHYA PRADESH', 'EAST RAJASTHAN',
             'EAST UTTAR PRADESH', 'GANGATIC WEST BENGAL', 'GUJARAT REGION', 'HARYANA DELHI & CHANDIGARH',
             'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'JHARKHAND', 'KERALA', 'KONKAN & GOA', 'LAKSHADWEEP',
             'MADHYA MAHARASHTRA', 'MATATHWADA', 'NAGA MANI MIZO TRIPURA', 'NORTH INTERIOR KARNATAKA',
             'ORISSA', 'PUNJAB', 'RAYALSEEMA', 'SAURASHTRA & KUTCH', 'SOUTH INTERIOR KARNATAKA',
             'SUB HIMALAYAN WEST BENGAL & SIKKIM', 'TAMIL NADU', 'TELANGANA', 'UTTARAKHAND', 'VIDARBHA',
             'WEST MADHYA PRADESH', 'WEST RAJASTHAN', 'WEST UTTAR PRADESH']
division = st.selectbox("Select Division", divisions)
input_data = pd.DataFrame([monthly_rainfall])
input_data[f"DIVISION_{division}"] = 1
for div in divisions:
    col = f"DIVISION_{div}"
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data.reindex(columns=feature_names, fill_value=0)
if st.button("Predict Annual Rainfall"):
    prediction = loaded_model.predict(input_data)[0]
    st.success(f"Predicted Annual Rainfall: {prediction:.2f} mm")
