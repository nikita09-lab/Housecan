
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load model
model = joblib.load('house_price_model.pkl')

st.header('Mumbai House Price Predictor')

# Load dataset for options
data = pd.read_csv('Ready_dataset.csv')

# INPUTS
house_type = st.selectbox(
    'Select House Type',
    sorted(data['type'].unique())
)

region = st.selectbox(
    'Select Region',
    sorted(data['region'].unique())
)

locality = st.selectbox(
    'Select Locality',
    sorted(data['locality'].unique())
)

bhk = st.selectbox(
    'Select BHK',
    sorted(data['bhk'].unique())
)

area = st.number_input('Enter total area (sqft)', min_value=100)

# Status mapping
status_map = {
    'Ready to Move': 0,
    'Under Construction': 1
}

status_label = st.selectbox(
    'Select House Status',
    list(status_map.keys())
)

status = status_map[status_label]

age = st.selectbox(
    'Select Age of House',
    sorted(data['age'].unique())
)

# Create input dataframe (MATCH TRAINING DATA)
test = pd.DataFrame(
    [[area, bhk, house_type, locality, region, status, age]],
    columns=['area', 'bhk', 'type', 'locality', 'region', 'status', 'age']
)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(test)
    st.success(f"Estimated House Price: ₹ {round(prediction[0], 2)}")
