"""
Concentration Tracker - Streamlit App
Simple web interface for mobile access via Streamlit Cloud
"""
import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import subprocess
import os
import platform
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Constants (same as Final.py)
CALIBRATION_DURATION = 8
EYE_CLOSED_TIME_LIMIT = 2.5
FACE_TILT_THRESHOLD_X = 0.04
FACE_TILT_THRESHOLD_Y = 0.05
CONCENTRATION_DROP_RATE = 3
CONCENTRATION_RECOVERY_RATE = 1.5
SMOOTHING_WINDOW = 8
EAR_DROP_THRESHOLD = 0.7

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_FACE = 234
RIGHT_FACE = 454

# Mediapipe Setup (will be initialized in callback)
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# Global face_mesh instance (thread-safe for WebRTC)
@st.cache_resource
def get_face_mesh():
    return mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

class SmoothingFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, value):
        self.values.append(value)
        return np.mean(self.values) if len(self.values) > 0 else value

def compute_EAR(landmarks, eye_indices, w, h):
    eye_points = []
    for idx in eye_indices:
        x = landmarks[idx].x * w
        y = landmarks[idx].y * h
        eye_points.append([x, y])
    eye_points = np.array(eye_points)
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

def compute_face_orientation(landmarks, w, h):
    nose = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
    chin = np.array([landmarks[CHIN].x, landmarks[CHIN].y])
    forehead = np.array([landmarks[FOREHEAD].x, landmarks[FOREHEAD].y])
    left_face = np.array([landmarks[LEFT_FACE].x, landmarks[LEFT_FACE].y])
    right_face = np.array([landmarks[RIGHT_FACE].x, landmarks[RIGHT_FACE].y])
    
    face_center_x = (left_face[0] + right_face[0]) / 2
    face_center_y = (forehead[1] + chin[1]) / 2
    
    horizontal_deviation = abs(nose[0] - face_center_x)
    vertical_deviation = abs(nose[1] - face_center_y)
    
    face_width = abs(right_face[0] - left_face[0])
    face_height = abs(chin[1] - forehead[1])
    
    norm_horizontal = horizontal_deviation / (face_width + 1e-6)
    norm_vertical = vertical_deviation / (face_height + 1e-6)
    
    return norm_horizontal, norm_vertical

def play_alert_sound():
    """Play alert sound using platform-specific command"""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.Popen(["afplay", "alert.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Linux":
            subprocess.Popen(["aplay", "alert.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Windows":
            subprocess.Popen(["powershell", "-c", "(New-Object Media.SoundPlayer 'alert.wav').PlaySync()"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass

# Initialize session state
if 'calibrating' not in st.session_state:
    st.session_state.calibrating = True
    st.session_state.calibration_start = time.time()
    st.session_state.concentration = 100.0
    st.session_state.ear_baseline = None
    st.session_state.ear_std = None
    st.session_state.adaptive_threshold = EAR_DROP_THRESHOLD
    st.session_state.ear_values = []
    st.session_state.neutral_face_x = None
    st.session_state.neutral_face_y = None
    st.session_state.face_orientations_x = []
    st.session_state.face_orientations_y = []
    st.session_state.session_start_time = None
    st.session_state.eyes_closed_start = None
    st.session_state.ear_filter = SmoothingFilter(SMOOTHING_WINDOW)
    st.session_state.concentration_filter = SmoothingFilter(SMOOTHING_WINDOW)
    st.session_state.face_tilt_x_filter = SmoothingFilter(SMOOTHING_WINDOW)
    st.session_state.face_tilt_y_filter = SmoothingFilter(SMOOTHING_WINDOW)
    st.session_state.prev_time = time.time()
    st.session_state.alert_played = False

def video_frame_callback(frame):
    try:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        current_time = time.time()
        
        # Get face_mesh instance
        face_mesh = get_face_mesh()
        
        # Flip for mirror effect
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        if st.session_state.calibrating:
            left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
            right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2
            st.session_state.ear_values.append(avg_ear)
            
            face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
            st.session_state.face_orientations_x.append(face_tilt_x)
            st.session_state.face_orientations_y.append(face_tilt_y)
            
            elapsed = current_time - st.session_state.calibration_start
            progress = min(1.0, elapsed / CALIBRATION_DURATION)
            
            if elapsed >= CALIBRATION_DURATION and len(st.session_state.ear_values) > 0:
                st.session_state.ear_baseline = np.mean(st.session_state.ear_values)
                st.session_state.ear_std = np.std(st.session_state.ear_values)
                st.session_state.adaptive_threshold = max(
                    st.session_state.ear_baseline * EAR_DROP_THRESHOLD,
                    st.session_state.ear_baseline - 2 * st.session_state.ear_std
                )
                st.session_state.neutral_face_x = np.mean(st.session_state.face_orientations_x)
                st.session_state.neutral_face_y = np.mean(st.session_state.face_orientations_y)
                st.session_state.calibrating = False
                st.session_state.session_start_time = current_time
                st.session_state.concentration = 100.0
                st.session_state.concentration_filter = SmoothingFilter(SMOOTHING_WINDOW)
                for _ in range(SMOOTHING_WINDOW):
                    st.session_state.concentration_filter.update(100.0)
            
            # Draw calibration UI
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            
            box_width, box_height = 400, 200
            box_x, box_y = (w - box_width) // 2, (h - box_height) // 2
            cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 200, 255), 3)
            
            title = "CALIBRATING..."
            cv2.putText(img, title, (box_x + 50, box_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
            bar_x, bar_y = box_x + 50, box_y + 100
            bar_width, bar_height = box_width - 100, 30
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
            fill_width = int(bar_width * progress)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (100, 200, 255), -1)
            
            progress_text = f"{int(progress * 100)}%"
            cv2.putText(img, progress_text, (box_x + 150, box_y + 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            time_remaining = max(0, CALIBRATION_DURATION - elapsed)
            time_text = f"{int(time_remaining)}s remaining"
            cv2.putText(img, time_text, (box_x + 80, box_y + 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            # Main tracking phase
            session_time = current_time - st.session_state.session_start_time
            
            left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
            right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2
            avg_ear = st.session_state.ear_filter.update(avg_ear)
            
            face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
            face_tilt_x = st.session_state.face_tilt_x_filter.update(face_tilt_x)
            face_tilt_y = st.session_state.face_tilt_y_filter.update(face_tilt_y)
            
            deviation_x = abs(face_tilt_x - st.session_state.neutral_face_x)
            deviation_y = abs(face_tilt_y - st.session_state.neutral_face_y)
            
            eyes_closed = avg_ear < st.session_state.adaptive_threshold
            
            dt = current_time - st.session_state.prev_time
            st.session_state.prev_time = current_time
            
            # Calculate concentration (same logic as Final.py)
            orientation_factor = 100.0
            if deviation_x > FACE_TILT_THRESHOLD_X or deviation_y > FACE_TILT_THRESHOLD_Y:
                max_deviation_ratio = max(deviation_x / FACE_TILT_THRESHOLD_X, deviation_y / FACE_TILT_THRESHOLD_Y)
                excess_deviation = max_deviation_ratio - 1.0
                
                if excess_deviation > 0.5:
                    effective_excess = excess_deviation - 0.5
                    
                    if effective_excess < 0.5:
                        penalty = effective_excess * 30
                    elif effective_excess < 1.5:
                        penalty = 15 + (effective_excess - 0.5) * 25
                    elif effective_excess < 2.5:
                        penalty = 40 + (effective_excess - 1.5) * 15
                    else:
                        penalty = 55 + min(20, (effective_excess - 2.5) * 0.5)
                    
                    orientation_factor = max(25, 100 - penalty)
            
            eye_factor = 100.0
            if eyes_closed:
                if st.session_state.eyes_closed_start is None:
                    st.session_state.eyes_closed_start = current_time
                closed_duration = current_time - st.session_state.eyes_closed_start
                if closed_duration > EYE_CLOSED_TIME_LIMIT:
                    penalty = min(50, closed_duration * CONCENTRATION_DROP_RATE * 0.8)
                    eye_factor = max(50, 100 - penalty)
            else:
                st.session_state.eyes_closed_start = None
                if orientation_factor >= 90:
                    eye_factor = min(100, st.session_state.concentration + CONCENTRATION_RECOVERY_RATE * dt)
                else:
                    eye_factor = 100.0
            
            ear_factor = 100.0
            if avg_ear < st.session_state.ear_baseline:
                ear_ratio = avg_ear / (st.session_state.ear_baseline + 1e-6)
                if ear_ratio < 0.85:
                    penalty = (0.85 - ear_ratio) * 40
                    ear_factor = max(60, 100 - penalty)
            
            st.session_state.concentration = 0.15 * eye_factor + 0.80 * orientation_factor + 0.05 * ear_factor
            st.session_state.concentration = st.session_state.concentration_filter.update(st.session_state.concentration)
            st.session_state.concentration = max(30, min(100, st.session_state.concentration))
            
            # Alert system
            if st.session_state.concentration < 50 and not st.session_state.alert_played:
                play_alert_sound()
                st.session_state.alert_played = True
            elif st.session_state.concentration >= 50:
                st.session_state.alert_played = False
            
            # Draw main UI
            bar_height = 60
            cv2.rectangle(img, (0, 0), (w, bar_height), (30, 30, 30), -1)
            
            concentration_text = f"{int(st.session_state.concentration)}%"
            if st.session_state.concentration >= 70:
                color = (0, 255, 100)
                status = "FOCUSED"
            elif st.session_state.concentration >= 50:
                color = (0, 200, 255)
                status = "MODERATE"
            else:
                color = (0, 100, 255)
                status = "DISTRACTED"
            
            cv2.putText(img, concentration_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(img, status, (150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            bar_x, bar_y = 10, 50
            bar_width = w - 20
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + 8), (50, 50, 50), -1)
            fill_width = int(bar_width * (st.session_state.concentration / 100))
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + 8), color, -1)
            
            time_text = f"Time: {int(session_time // 60):02d}:{int(session_time % 60):02d}"
            cv2.putText(img, time_text, (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            if st.session_state.concentration < 50:
                flash_speed = 6
                alpha = 0.4 + 0.4 * np.sin(time.time() * flash_speed)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                
                warning_text = "PLEASE FOCUS!"
                warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
                cv2.putText(img, warning_text, ((w - warning_size[0]) // 2 + 2, h // 2 + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
                cv2.putText(img, warning_text, ((w - warning_size[0]) // 2, h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Draw face landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image=img,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec,
        )
    else:
        # No face detected
        if not st.session_state.calibrating:
            st.session_state.concentration = max(30, st.session_state.concentration - 5)
            if st.session_state.concentration < 50 and not st.session_state.alert_played:
                play_alert_sound()
                st.session_state.alert_played = True
            cv2.putText(img, "NO FACE DETECTED", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        # Return original frame if processing fails
        print(f"Error in video callback: {e}")
        return frame

# Streamlit UI
st.set_page_config(page_title="Concentration Tracker", page_icon="📱", layout="wide")

st.title("📱 Concentration Tracker")
st.markdown("### Real-time focus monitoring using your camera")

# Sidebar
with st.sidebar:
    st.header("Status")
    if st.session_state.calibrating:
        st.info("🔄 Calibrating...")
        elapsed = time.time() - st.session_state.calibration_start
        progress = min(1.0, elapsed / CALIBRATION_DURATION)
        st.progress(progress)
        st.caption(f"{int((CALIBRATION_DURATION - elapsed))}s remaining")
    else:
        concentration = int(st.session_state.concentration)
        if concentration >= 70:
            st.success(f"✅ Concentration: {concentration}%")
        elif concentration >= 50:
            st.warning(f"⚠️ Concentration: {concentration}%")
        else:
            st.error(f"❌ Concentration: {concentration}%")
        
        if st.session_state.session_start_time:
            session_time = time.time() - st.session_state.session_start_time
            st.caption(f"Session time: {int(session_time // 60):02d}:{int(session_time % 60):02d}")
    
    st.divider()
    if st.button("🔄 Reset Session"):
        st.session_state.calibrating = True
        st.session_state.calibration_start = time.time()
        st.session_state.concentration = 100.0
        st.session_state.ear_values = []
        st.session_state.face_orientations_x = []
        st.session_state.face_orientations_y = []
        st.session_state.alert_played = False
        st.session_state.ear_filter = SmoothingFilter(SMOOTHING_WINDOW)
        st.session_state.concentration_filter = SmoothingFilter(SMOOTHING_WINDOW)
        st.rerun()

# WebRTC streamer
st.markdown("---")
st.subheader("📹 Camera Feed")

webrtc_ctx = webrtc_streamer(
    key="concentration-tracker",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "facingMode": "user"
        },
        "audio": False
    },
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("✅ Camera is active! Make sure you're facing the camera.")
    st.info("💡 **Tip:** If you don't see the camera, check your browser's camera permissions.")
elif webrtc_ctx.state.playing is None:
    st.warning("⏳ Click 'START' above to begin camera feed. Make sure to allow camera access when prompted.")
else:
    st.error("❌ Camera not available. Please check your camera permissions and try again.")

