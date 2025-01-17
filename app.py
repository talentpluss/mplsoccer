import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os

# Page Configuration
st.set_page_config(page_title="YOLO Soccer Movement", layout="wide")

# Tabs for the application
tabs = ["Upload Video", "Single Video Output", "All Videos Output"]
selected_tab = st.sidebar.radio("Navigation", tabs)

if selected_tab == "Upload Video":
    st.title("Upload and Process Video")
    uploaded_video = st.file_uploader("Upload a Soccer Video", type=["mp4", "avi"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.success("Video uploaded successfully!")
        
        # Placeholder for processing logic
        st.info("Video processing feature is under construction.")

elif selected_tab == "Single Video Output":
    st.title("Single Video Output")

    # File uploader for CSV
    uploaded_csv = st.file_uploader("Upload CSV for Single Video", type=["csv"])

    if uploaded_csv is not None:
        # Load data
        df = pd.read_csv(uploaded_csv)
        st.write("Preview of the data:", df.head())

        # Normalize data for pitch
        width, height = 120, 80  # Dimensions of the pitch in meters
        df["pitch_x"] = df["centroid_x"] / df["centroid_x"].max() * width
        df["pitch_y"] = df["centroid_y"] / df["centroid_y"].max() * height

        # Plot using mplsoccer
        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        fig, ax = plt.subplots(figsize=(12, 8))
        pitch.draw(ax=ax)
        pitch.scatter(df["pitch_x"], df["pitch_y"], ax=ax, s=30, c='red', edgecolors='black')

        st.pyplot(fig)

elif selected_tab == "All Videos Output":
    st.title("All Videos Output")

    # File uploader for CSV
    uploaded_csv = st.file_uploader("Upload CSV for All Videos", type=["csv"])

    if uploaded_csv is not None:
        # Load data
        df = pd.read_csv(uploaded_csv)
        st.write("Preview of the data:", df.head())

        # Normalize data for pitch
        width, height = 120, 80  # Dimensions of the pitch in meters
        df["pitch_x"] = df["centroid_x"] / df["centroid_x"].max() * width
        df["pitch_y"] = df["centroid_y"] / df["centroid_y"].max() * height

        # Plot using mplsoccer
        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        fig, ax = plt.subplots(figsize=(12, 8))
        pitch.draw(ax=ax)
        pitch.scatter(df["pitch_x"], df["pitch_y"], ax=ax, s=30, c='blue', edgecolors='black')

        st.pyplot(fig)

else:
    st.error("Invalid tab selected.")

