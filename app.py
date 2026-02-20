import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle

# 1. Page Config & Title
st.set_page_config(page_title="AI Fitness Coach", layout="wide")
st.title("AI Virtual Fitness Coach 🏋️‍♂️")

# 2. Setup MediaPipe ONCE (Outside the loop for speed)
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create the landmarker instance globally to prevent lag
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)
landmarker = PoseLandmarker.create_from_options(options)

# Use a list for the counter so it can be updated inside the thread
counter_container = {"count": 0, "stage": None}

# 3. Define the Callback Function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Flip image for "Mirror" effect (feels more natural)
    img = cv2.flip(img, 1)
    
    # Process with MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        
        # Get Shoulder, Elbow, Wrist
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # Rep counting logic
        if angle > 160: 
            counter_container["stage"] = "down"
        if angle < 30 and counter_container["stage"] == "down":
            counter_container["stage"] = "up"
            counter_container["count"] += 1

        # Draw Feedback on image
        h, w, _ = img.shape
        cv2.putText(img, f"Angle: {int(angle)}", (int(elbow[0]*w), int(elbow[1]*h) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # UI Overlay: Purple box for Reps
    cv2.rectangle(img, (0, 0), (250, 80), (128, 0, 128), -1)
    cv2.putText(img, f"REPS: {counter_container['count']}", (20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Streamer with Optimized Settings
webrtc_streamer(
    key="fitness-coach",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640}, 
            "height": {"ideal": 480},
            "frameRate": {"ideal": 20}
        },
        "audio": False
    },
    async_processing=True, # Critical for preventing the "pause"
)

st.info("💡 Tip: Step back so your full upper body is visible. Keep your elbow steady while curling!")