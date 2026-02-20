import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
from utils import calculate_angle

# 1. Streamlit UI Configuration
st.set_page_config(page_title="AI Virtual Fitness Coach", layout="wide")
st.title("AI Virtual Fitness Coach 🏋️‍♂️")

# Sidebar for controls and stats
st.sidebar.title("Workout Dashboard")
exercise = st.sidebar.selectbox("Select Exercise", ["Bicep Curls"])
target_goal = st.sidebar.number_input("Set Rep Goal", min_value=1, value=10)

# 2. Setup MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global state for Streamlit
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'stage' not in st.session_state:
    st.session_state.stage = None

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

# 3. Video Processing Loop
FRAME_WINDOW = st.image([]) # Placeholder for the webcam feed
cap = cv2.VideoCapture(0, cv2.CAP_MSMF) # Using stable Windows backend

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not detected.")
            break

        # Convert frame for MediaPipe and Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        if latest_result and latest_result.pose_landmarks:
            landmarks = latest_result.pose_landmarks[0]
            
            # Use landmarks 11, 13, 15 for left arm
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Rep Logic
            if angle > 160:
                st.session_state.stage = "down"
            if angle < 30 and st.session_state.stage == "down":
                st.session_state.stage = "up"
                st.session_state.counter += 1

        # 4. Update the Web UI
        # Draw counter on the frame
        cv2.putText(frame_rgb, f"Reps: {st.session_state.counter}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Update sidebar metrics
        st.sidebar.metric("Repetitions", st.session_state.counter)
        progress = min(st.session_state.counter / target_goal, 1.0)
        st.sidebar.progress(progress)
        
        # Display the frame in Streamlit
        FRAME_WINDOW.image(frame_rgb)

cap.release()