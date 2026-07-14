import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

# --- 1. UI SETUP ---
st.set_page_config(page_title="Peace Blur", layout="centered")

st.markdown("""
    <style>
    .stApp { background: #121212; color: white; }
    .glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    </style>
    <div class="glass">
        <h1>✌️ Peace Blur App</h1>
        <p>Angkat jari telunjuk & tengah (Peace Sign) untuk nge-blur layar.</p>
    </div>
""", unsafe_allow_html=True)

# --- 2. MEDIAPIPE MODEL AUTO-DOWNLOAD ---
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Setup Task API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. PROCESSING LOGIC ---
class GestureTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = detector.detect(mp_image)
        blur_active = False
        
        if detection_result.hand_landmarks:
            for landmarks in detection_result.hand_landmarks:
                # Cek jari (Peace Gesture: Index & Middle UP, Ring & Pinky DOWN)
                if (landmarks[8].y < landmarks[6].y) and (landmarks[12].y < landmarks[10].y) and \
                   (landmarks[16].y > landmarks[14].y) and (landmarks[20].y > landmarks[18].y):
                    blur_active = True

                # Titik hijau transparan (landmark)
                overlay = img.copy()
                for lm in landmarks:
                    cv2.circle(overlay, (int(lm.x * img.shape[1]), int(lm.y * img.shape[0])), 3, (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        if blur_active:
            img = cv2.GaussianBlur(img, (99, 99), 0)
        
        return img

# --- 4. STREAMER ---
webrtc_streamer(
    key="blur-app",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=GestureTransformer,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)