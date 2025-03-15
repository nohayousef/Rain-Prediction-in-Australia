import streamlit as st
#from transformers import pipeline
import streamlit as st
import pandas as pd
import os
import sys

import joblib  # Corrected model loading
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
# # ----------------- Load Dataset Safely ----------------- #
# Date Input
date = st.date_input("Select Date")
day = float(date.day)
month = float(date.month)

# Numerical Inputs
minTemp = st.number_input("Min Temperature (°C)", value=0.0)
maxTemp = st.number_input("Max Temperature (°C)", value=0.0)
rainfall = st.number_input("Rainfall (mm)", value=0.0)
evaporation = st.number_input("Evaporation", value=0.0)
sunshine = st.number_input("Sunshine Hours", value=0.0)
windGustSpeed = st.number_input("Wind Gust Speed (km/h)", value=0.0)
windSpeed9am = st.number_input("Wind Speed 9AM (km/h)", value=0.0)
windSpeed3pm = st.number_input("Wind Speed 3PM (km/h)", value=0.0)
humidity9am = st.number_input("Humidity 9AM (%)", value=0.0)
humidity3pm = st.number_input("Humidity 3PM (%)", value=0.0)
pressure9am = st.number_input("Pressure 9AM (hPa)", value=0.0)
pressure3pm = st.number_input("Pressure 3PM (hPa)", value=0.0)
temp9am = st.number_input("Temperature 9AM (°C)", value=0.0)
temp3pm = st.number_input("Temperature 3PM (°C)", value=0.0)
cloud9am = st.number_input("Cloud Cover 9AM", value=0.0)
cloud3pm = st.number_input("Cloud Cover 3PM", value=0.0)

# Categorical Inputs
location = st.selectbox("Location", options=list(range(1, 50)))  # Example: 1-50 locations
windDir9am = st.selectbox("Wind Direction 9AM", options=list(range(1, 17)))  # Example: 16 directions
windDir3pm = st.selectbox("Wind Direction 3PM", options=list(range(1, 17)))
windGustDir = st.selectbox("Wind Gust Direction", options=list(range(1, 17)))
rainToday = st.radio("Did it rain today?", options=[0, 1])

# Prediction Button
if st.button("Predict Weather"):
    # Prepare input data
    input_lst = [
        location, minTemp, maxTemp, rainfall, evaporation, sunshine,
        windGustDir, windGustSpeed, windDir9am, windDir3pm, windSpeed9am, windSpeed3pm,
        humidity9am, humidity3pm, pressure9am, pressure3pm, cloud9am, cloud3pm,
        temp9am, temp3pm, rainToday, month, day
    ]

    # Convert to NumPy array & reshape for model
    input_array = np.array(input_lst).reshape(1, -1)

    # Make prediction
    # ----------------- Function to Load Model ----------------- #
@st.cache_resource  # Cache model to improve performance
def load_model():
    try:
        with open("/content/voting_pipeline.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'voting_pipeline.pkl' exists.")
        return None

    pred = model.predict(input_array)[0]

    # Display result
    if pred == 0:
        st.success("☀️ The weather is predicted to be **Sunny**!")
    else:
        st.warning("🌧️ The weather is predicted to be **Rainy**!")

