import streamlit as st
#from transformers import pipeline
import sys
import os
import streamlit as st
import pandas as pd
import os
import pickle
import plotly.express as px

# ----------------- Constants ----------------- #
MODEL_FILE = "/content/voting_pipeline.pkl"
DATA_FILE = "/content/weatherAUS.csv"
IMAGE_FILE = "/content/Screenshot 2025-03-09 201614.png"

# ----------------- Page Configuration ----------------- #
st.set_page_config(page_title="Rain Prediction", layout="wide")

# ----------------- Function to Load Model ----------------- #
@st.cache_resource
def load_model():
    """Loads and caches the machine learning model."""
    if not os.path.exists(MODEL_FILE):
        st.error("❌ Model file not found. Please ensure 'voting_pipeline.pkl' exists.")
        return None
    try:
        with open(MODEL_FILE, "rb") as file:
            model = pickle.load(file, fix_imports=True, encoding="latin1")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

# ----------------- Function to Load Dataset ----------------- #
@st.cache_data
def load_data():
    """Loads and caches the dataset."""
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ Dataset '{DATA_FILE}' not found.")
        return None
    try:
        return pd.read_csv(DATA_FILE, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        st.error(f"⚠️ Failed to load dataset: {str(e)}")
        return None

# ----------------- Load Model & Dataset ----------------- #
model = load_model()
df_ = load_data()

# ----------------- Sidebar Navigation ----------------- #
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Exploration", "🔮 Predict Rain"])

# ----------------- Home Page ----------------- #
if page == "🏠 Home":
    st.title("🌦️ Rain Prediction in Australia")
    st.header("Predict Next-Day Rain in Australia")
    st.write("""
    Ever wondered if you should carry an umbrella tomorrow?  
    This dataset allows you to train classification models to predict next-day rain using the target variable **RainTomorrow**.
    """)

    # Display Image
    if os.path.exists(IMAGE_FILE):
        st.image(IMAGE_FILE, caption="Weather Data Visualization", width=500)
    else:
        st.warning("⚠️ Image not found.")

    # Dataset Overview
    st.subheader("📌 About the Dataset")
    st.write("""
    This dataset contains about **10 years** of daily weather observations from numerous Australian weather stations.

    - **RainTomorrow** is the target variable to predict.
    - It indicates whether it rained the next day (**Yes** or **No**).
    - The column is **Yes** if the rain for that day was **1mm or more**.
    """)

# ----------------- Data Exploration Page ----------------- #
elif page == "📊 Data Exploration":
    st.title("📊 Explore Weather Data")
    
    if df_ is not None:
        st.write(f"✅ Loaded {df_.shape[0]:,} rows and {df_.shape[1]:,} columns.")

        # Show dataset preview
        st.subheader("🔍 Preview of the Data")
        st.dataframe(df_.head())

        # Show dataset summary
        st.subheader("📌 Dataset Summary")
        st.write(df_.describe())

        # Plot a distribution of Rainfall
        st.subheader("🌧️ Rainfall Distribution")
        fig = px.histogram(df_, x="Rainfall", nbins=50, title="Distribution of Rainfall")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Data not loaded. Please check the dataset file.")

# ----------------- Prediction Page ----------------- #
elif page == "🔮 Predict Rain":
    st.title("🔮 Rain Prediction")
    
    if model is not None:
        st.write("✅ Model loaded successfully. Ready to make predictions.")
        # Placeholder for prediction functionality
        st.warning("⚠️ Prediction input form will be added here.")
    else:
        st.error("❌ Model not loaded. Please check the model file.")
