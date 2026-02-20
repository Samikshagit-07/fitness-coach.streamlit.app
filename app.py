import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle

# 1. Page Config
st.set_page_config(page_title="AI Fitness Coach", layout="wide")
st.title("AI Virtual Fitness Coach 🏋️‍♂️")

# 2. Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global-like state for the counter (Thread-safe)
if 'count' not in st.session_state:
    st.session_state['count'] = 0
    st.session_state['stage'] = None

# 3. Define the Callback Function (This processes every frame)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Initialize Landmarker inside callback or use a shared instance
    # For simplicity in this demo, we'll use a local instance
    with PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE # Using IMAGE mode for frame-by-frame
        )
    ) as landmarker:
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            # Indices: 11(Shoulder), 13(Elbow), 15(Wrist)
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Rep counting logic
            if angle > 160: st.session_state['stage'] = "down"
            if angle < 30 and st.session_state['stage'] == "down":
                st.session_state['stage'] = "up"
                st.session_state['count'] += 1

            # Draw Feedback
            h, w, _ = img.shape
            cv2.putText(img, str(int(angle)), (int(elbow[0]*w), int(elbow[1]*h)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # UI Overlay on frame
    cv2.rectangle(img, (0,0), (200,80), (128, 0, 128), -1)
    cv2.putText(img, f"REPS: {st.session_state['count']}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Start the WebRTC Streamer
webrtc_streamer(
    key="fitness-coach",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)

st.write(f"### Total Repetitions: {st.session_state['count']}")