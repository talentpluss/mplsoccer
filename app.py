import streamlit as st
        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        fig, ax = pitch.draw(figsize=(12, 8))
        pitch.scatter(trajectories["pitch_x"], trajectories["pitch_y"], ax=ax, s=30, c='red', edgecolors='black')
        st.pyplot(fig)

# Tab 2: Single Video Analysis
with tabs[1]:
    st.header("Single Video Analysis")
    uploaded_csv = st.file_uploader("Upload a CSV File", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write(df.head())

        df["pitch_x"] = df["centroid_x"] / 1920 * 120  # Adjust width as per video dimensions
        df["pitch_y"] = df["centroid_y"] / 1080 * 80   # Adjust height as per video dimensions

        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        fig, ax = pitch.draw(figsize=(12, 8))
        pitch.scatter(df["pitch_x"], df["pitch_y"], ax=ax, s=30, c='blue', edgecolors='black')
        st.pyplot(fig)

# Tab 3: Combined Analysis
with tabs[2]:
    st.header("Combined Video Analysis")
    combined_csv = st.file_uploader("Upload Combined CSV File", type=["csv"])

    if combined_csv:
        df = pd.read_csv(combined_csv)
        st.write(df.head())

        df["pitch_x"] = df["centroid_x"] / 1920 * 120
        df["pitch_y"] = df["centroid_y"] / 1080 * 80

        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        fig, ax = pitch.draw(figsize=(12, 8))
        pitch.scatter(df["pitch_x"], df["pitch_y"], ax=ax, s=20, c='green', edgecolors='black')
        st.pyplot(fig)
