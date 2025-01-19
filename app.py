import streamlit as st
import pandas as pd
import numpy as np
import cv2
from mplsoccer import Pitch
from ultralytics import YOLO
import tempfile
import os

# Initialize YOLO model
model = YOLO("yolov5s.pt")  # Replace with custom model if needed

# Streamlit App
st.set_page_config(page_title="Player Movement Analysis", layout="wide")
st.title("Player Movement Analysis App")

# Tabs for the interface
tabs = st.tabs(["Upload and Process Video", "Single Video Analysis", "Combined Analysis"])

# Tab 1: Upload and Process Video
with tabs[0]:
    st.header("Upload and Process Video")

    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        # Save uploaded video to a temporary file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(video_path)
        st.write("Processing video...")

        # Read the video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform YOLO detection on the frame
            results = model(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2, conf, cls = box.xyxy[0]
                    detections.append({
                        "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        "x1": x1.item(),
                        "y1": y1.item(),
                        "x2": x2.item(),
                        "y2": y2.item(),
                        "confidence": conf.item(),
                        "class": cls.item()
                    })

        cap.release()

        # Convert detections to a DataFrame
        detections_df = pd.DataFrame(detections)

        # Calculate centroids for each detection
        detections_df["centroid_x"] = (detections_df["x1"] + detections_df["x2"]) / 2
        detections_df["centroid_y"] = (detections_df["y1"] + detections_df["y2"]) / 2

        # Map centroids to soccer pitch dimensions
        trajectories = detections_df.groupby("frame")[["centroid_x", "centroid_y"]].mean().reset_index()
        trajectories["pitch_x"] = trajectories["centroid_x"] / width * 120  # Adjust width to 120 meters
        trajectories["pitch_y"] = trajectories["centroid_y"] / height * 80  # Adjust height to 80 meters

        # Visualize centroids on soccer pitch
        st.write("Soccer Field Visualization:")
        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        fig, ax = pitch.draw(figsize=(12, 8))
        pitch.scatter(trajectories["pitch_x"], trajectories["pitch_y"], ax=ax, s=30, c='red', edgecolors='black')
        st.pyplot(fig)

# Tab 2: Single Video Analysis
with tabs[1]:
    # Tab 2: Single Video Analysis
    st.header("Single Video Analysis")
    
    # Read the sample CSV file directly
    csv_file_path = "sampel.csv"  # Pastikan file ini berada di lokasi yang sama dengan script
    df = pd.read_csv(csv_file_path)
    
    # Display the first few rows of the dataset
    st.write("Sample Data Preview:")
    st.write(df.head())
    
    # Calculate pitch coordinates
    df["pitch_x"] = df["centroid_x"] / 1920 * 120  # Adjust width as per video dimensions
    df["pitch_y"] = df["centroid_y"] / 1080 * 80   # Adjust height as per video dimensions
    
    # Plotting on a soccer pitch
    st.write("Soccer Field Visualization:")
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 8))
    pitch.scatter(df["pitch_x"], df["pitch_y"], ax=ax, s=30, c='blue', edgecolors='black')
    
    # Display the plot
    st.pyplot(fig)

# Tab 3: Combined Analysis
with tabs[2]:
    # Tab 3: Combined Video Analysis
    st.header("Combined Video Analysis")
    
    # Read the combined CSV file directly
    csv_file_path = "allvideo.csv"  # Pastikan file ini berada di lokasi yang sama dengan script
    df = pd.read_csv(csv_file_path)
    
    # Display the first few rows of the dataset
    st.write("Combined Data Preview:")
    st.write(df.head())
    
    # Calculate pitch coordinates
    df["pitch_x"] = df["centroid_x"] / 1920 * 120  # Adjust width as per video dimensions
    df["pitch_y"] = df["centroid_y"] / 1080 * 80   # Adjust height as per video dimensions
    
    # Plotting on a soccer pitch
    st.write("Soccer Field Visualization:")
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 8))
    pitch.scatter(df["pitch_x"], df["pitch_y"], ax=ax, s=20, c='green', edgecolors='black')
    
    # Display the plot
    st.pyplot(fig)
