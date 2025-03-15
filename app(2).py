import streamlit as st
#from transformers import pipeline
import sys
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- Load Dataset ----------------- #
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/content/weatherAUS.csv", encoding="utf-8", on_bad_lines="skip")
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to load dataset: {str(e)}")
        return None

df = load_data()

# ----------------- Data Visualization Section ----------------- #
st.title("📊 Weather Data Visualization")

if df is not None:
    # Create a sidebar selection for visualization options
    visualization_option = st.sidebar.selectbox(
        "Select a Visualization Type",
        [
            "Rain Probability by Wind Direction",
            "Heatmap of Correlation Matrix",
            "Previous Day Rainfall vs. Rain Tomorrow",
            "Seasonal Effect on Rain Tomorrow"
        ]
    )

    # -------------- Rain Probability by WindGustDir -------------- #
    if visualization_option == "Rain Probability by Wind Direction":
        st.subheader("🌬️ Rain Probability by Wind Gust Direction")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=df["WindGustDir"], y=df["RainTomorrow"], ax=ax, palette="coolwarm")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title("Rain Probability by WindGustDir", fontsize=14)
        st.pyplot(fig)

    # -------------- Heatmap of Correlation Matrix -------------- #
    elif visualization_option == "Heatmap of Correlation Matrix":
        st.subheader("🔥 Correlation Heatmap")
        numeric_df = df.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # -------------- Previous Day Rainfall vs. Rain Tomorrow -------------- #
    elif visualization_option == "Previous Day Rainfall vs. Rain Tomorrow":
        st.subheader("🌧️ Previous Day Rainfall vs. Rain Tomorrow")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="RainTomorrow", y="PrevDayRainfall", palette=["blue", "red"], ax=ax)
        ax.set_title("Previous Day Rainfall vs. Rain Tomorrow", fontsize=14)
        ax.set_xlabel("Rain Tomorrow (No=0, Yes=1)")
        ax.set_ylabel("Mean Previous Day Rainfall (mm)")
        st.pyplot(fig)

    # -------------- Seasonal Effect on Rain Tomorrow -------------- #
    elif visualization_option == "Seasonal Effect on Rain Tomorrow":
        st.subheader("🌦️ Seasonal Effect on Rain Tomorrow")

        # Define function to map months to seasons
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        df["Season"] = df["Date"].dt.month.apply(get_season)
        season_counts = df.groupby("Season")["RainTomorrow"].value_counts(normalize=True).unstack()

        fig, ax = plt.subplots(figsize=(8, 5))
        season_counts.plot(kind="bar", stacked=True, colormap="coolwarm", alpha=0.8, ax=ax)
        ax.set_title("Seasonal Effect on Rain Tomorrow", fontsize=14, fontweight="bold")
        ax.set_xlabel("Season", fontsize=12)
        ax.set_ylabel("Proportion", fontsize=12)
        ax.legend(["No Rain", "Rain"], title="RainTomorrow")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)

else:
    st.error("❌ Data not loaded. Please check the dataset file.")
