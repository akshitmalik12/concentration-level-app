"""
Concentration Tracker - Android Version
Built with Kivy for Android deployment
"""
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.logger import Logger
import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import os

# Android-specific imports (only available on Android)
try:
    from jnius import autoclass
    from android.permissions import request_permissions, Permission
    PythonActivity = autoclass('org.kivy.android.PythonActivity')
    MediaPlayer = autoclass('android.media.MediaPlayer')
    Uri = autoclass('android.net.Uri')
    Context = autoclass('android.content.Context')
    File = autoclass('java.io.File')
    ANDROID_AVAILABLE = True
except ImportError:
    # Not on Android - will be available when built for Android
    ANDROID_AVAILABLE = False
    PythonActivity = None
    MediaPlayer = None
    File = None

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

# Face orientation landmarks
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_FACE = 234
RIGHT_FACE = 454

class SmoothingFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, value):
        self.values.append(value)
        return np.mean(self.values) if len(self.values) > 0 else value
    
    def reset(self):
        self.values.clear()

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

def play_alert_android():
    """Play alert sound on Android"""
    if not ANDROID_AVAILABLE:
        Logger.warning("Alert: Android modules not available (not on Android)")
        return
    
    try:
        activity = PythonActivity.mActivity
        # Try multiple paths for the sound file
        sound_paths = [
            os.path.join(os.path.dirname(__file__), 'alert.wav'),
            'alert.wav',
            os.path.join(activity.getFilesDir().getPath(), 'alert.wav'),
        ]
        
        # Try to get from assets first (Android assets)
        sound_path = None
        try:
            if ANDROID_AVAILABLE:
                AssetManager = autoclass('android.content.res.AssetManager')
                asset_manager = activity.getAssets()
                # Try to open from assets
                sound_file = asset_manager.open('alert.wav')
                # If we can open it, copy to internal storage and play
                internal_path = os.path.join(activity.getFilesDir().getPath(), 'alert.wav')
                with open(internal_path, 'wb') as f:
                    f.write(sound_file.read())
                sound_file.close()
                sound_path = internal_path
        except Exception as asset_error:
            Logger.debug(f"Could not load from assets: {asset_error}")
            # Fallback to file system
            for path in sound_paths:
                if os.path.exists(path):
                    sound_path = path
                    break
        
        if sound_path and os.path.exists(sound_path):
            media_player = MediaPlayer()
            # Use file:// URI for local files
            if sound_path.startswith('/'):
                file_obj = File(sound_path)
                uri = Uri.fromFile(file_obj)
            else:
                uri = Uri.parse(sound_path)
            media_player.setDataSource(activity, uri)
            media_player.prepare()
            media_player.start()
            Logger.info("Alert: Sound playing")
        else:
            Logger.warning(f"Alert: Sound file not found in any location: {sound_paths}")
    except Exception as e:
        Logger.error(f"Alert: Error playing sound: {e}")
        import traceback
        Logger.error(traceback.format_exc())

class ConcentrationTrackerApp(App):
    def build(self):
        # Request camera permission (Android only)
        if ANDROID_AVAILABLE:
            try:
                request_permissions([Permission.CAMERA, Permission.RECORD_AUDIO])
            except:
                pass
        
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Video display
        self.image = Image()
        layout.add_widget(self.image)
        
        # Status label
        self.status_label = Label(
            text='Initializing...',
            size_hint_y=None,
            height=50,
            font_size=20
        )
        layout.add_widget(self.status_label)
        
        # Concentration label
        self.concentration_label = Label(
            text='Concentration: 100%',
            size_hint_y=None,
            height=50,
            font_size=24,
            bold=True
        )
        layout.add_widget(self.concentration_label)
        
        # Control buttons
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        
        self.reset_btn = Button(text='Reset', on_press=self.reset_session)
        self.pause_btn = Button(text='Pause', on_press=self.toggle_pause)
        
        button_layout.add_widget(self.reset_btn)
        button_layout.add_widget(self.pause_btn)
        layout.add_widget(button_layout)
        
        # Initialize tracking
        self.init_tracking()
        
        # Schedule frame updates
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 30 FPS
        
        return layout
    
    def init_tracking(self):
        """Initialize tracking variables"""
        self.cap = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        
        self.calibrating = True
        self.calibration_start = time.time()
        self.concentration = 100.0
        self.eyes_closed_start = None
        self.ear_baseline = None
        self.ear_std = None
        self.adaptive_threshold = EAR_DROP_THRESHOLD
        self.ear_values = []
        self.neutral_face_x = None
        self.neutral_face_y = None
        self.face_orientations_x = []
        self.face_orientations_y = []
        self.session_start_time = None
        self.ear_filter = SmoothingFilter(SMOOTHING_WINDOW)
        self.concentration_filter = SmoothingFilter(SMOOTHING_WINDOW)
        self.face_tilt_x_filter = SmoothingFilter(SMOOTHING_WINDOW)
        self.face_tilt_y_filter = SmoothingFilter(SMOOTHING_WINDOW)
        self.alert_played = False
        self.prev_time = time.time()
        self.paused = False
        
        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                Logger.error("Could not open camera")
                self.status_label.text = "Error: Could not open camera"
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            Logger.error(f"Camera error: {e}")
            self.status_label.text = f"Camera error: {e}"
    
    def update_frame(self, dt):
        """Update video frame and process"""
        if self.paused or self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            if self.calibrating:
                # Calibration phase
                left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
                right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2
                self.ear_values.append(avg_ear)
                
                face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
                self.face_orientations_x.append(face_tilt_x)
                self.face_orientations_y.append(face_tilt_y)
                
                elapsed = current_time - self.calibration_start
                progress = min(1.0, elapsed / CALIBRATION_DURATION)
                
                if elapsed >= CALIBRATION_DURATION and len(self.ear_values) > 0:
                    self.ear_baseline = np.mean(self.ear_values)
                    self.ear_std = np.std(self.ear_values)
                    self.adaptive_threshold = max(
                        self.ear_baseline * EAR_DROP_THRESHOLD,
                        self.ear_baseline - 2 * self.ear_std
                    )
                    self.neutral_face_x = np.mean(self.face_orientations_x)
                    self.neutral_face_y = np.mean(self.face_orientations_y)
                    self.calibrating = False
                    self.session_start_time = current_time
                    self.concentration = 100.0
                    self.concentration_filter = SmoothingFilter(SMOOTHING_WINDOW)
                    for _ in range(SMOOTHING_WINDOW):
                        self.concentration_filter.update(100.0)
                
                # Draw calibration UI
                self.draw_calibration_ui(frame, progress, w, h)
            else:
                # Main tracking phase
                session_time = current_time - self.session_start_time
                
                left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
                right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2
                avg_ear = self.ear_filter.update(avg_ear)
                
                face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
                face_tilt_x = self.face_tilt_x_filter.update(face_tilt_x)
                face_tilt_y = self.face_tilt_y_filter.update(face_tilt_y)
                
                deviation_x = abs(face_tilt_x - self.neutral_face_x)
                deviation_y = abs(face_tilt_y - self.neutral_face_y)
                
                eyes_closed = avg_ear < self.adaptive_threshold
                
                dt = current_time - self.prev_time
                self.prev_time = current_time
                
                # Calculate concentration (same logic as Final.py with reduced penalties)
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
                    if self.eyes_closed_start is None:
                        self.eyes_closed_start = current_time
                    closed_duration = current_time - self.eyes_closed_start
                    if closed_duration > EYE_CLOSED_TIME_LIMIT:
                        penalty = min(50, closed_duration * CONCENTRATION_DROP_RATE * 0.8)
                        eye_factor = max(50, 100 - penalty)
                else:
                    self.eyes_closed_start = None
                    if orientation_factor >= 90:
                        eye_factor = min(100, self.concentration + CONCENTRATION_RECOVERY_RATE * dt)
                    else:
                        eye_factor = 100.0
                
                ear_factor = 100.0
                if avg_ear < self.ear_baseline:
                    ear_ratio = avg_ear / (self.ear_baseline + 1e-6)
                    if ear_ratio < 0.85:
                        penalty = (0.85 - ear_ratio) * 40
                        ear_factor = max(60, 100 - penalty)
                
                self.concentration = 0.15 * eye_factor + 0.80 * orientation_factor + 0.05 * ear_factor
                self.concentration = self.concentration_filter.update(self.concentration)
                self.concentration = max(30, min(100, self.concentration))
                
                # Alert system
                if self.concentration < 50 and not self.alert_played:
                    Clock.schedule_once(lambda dt: play_alert_android(), 0)
                    self.alert_played = True
                elif self.concentration >= 50:
                    self.alert_played = False
                
                # Draw main UI
                self.draw_main_ui(frame, self.concentration, avg_ear, face_tilt_x, face_tilt_y,
                                 eyes_closed, session_time, w, h)
                
                # Update labels
                self.concentration_label.text = f'Concentration: {int(self.concentration)}%'
                if self.concentration >= 70:
                    self.concentration_label.color = (0, 1, 0)
                    self.status_label.text = 'FOCUSED'
                elif self.concentration >= 50:
                    self.concentration_label.color = (1, 0.8, 0)
                    self.status_label.text = 'MODERATE'
                else:
                    self.concentration_label.color = (1, 0, 0)
                    self.status_label.text = 'DISTRACTED - PLEASE FOCUS!'
            
            # Draw face landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec,
            )
        else:
            # No face detected
            if not self.calibrating:
                self.concentration = max(30, self.concentration - 5)
                if self.concentration < 50 and not self.alert_played:
                    Clock.schedule_once(lambda dt: play_alert_android(), 0)
                    self.alert_played = True
                cv2.putText(frame, "NO FACE DETECTED", (w//2 - 150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert to texture for Kivy
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(w, h), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.image.texture = texture
    
    def draw_calibration_ui(self, frame, progress, w, h):
        """Draw calibration UI"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        box_width, box_height = 400, 200
        box_x, box_y = (w - box_width) // 2, (h - box_height) // 2
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 200, 255), 3)
        
        title = "CALIBRATING..."
        cv2.putText(frame, title, (box_x + 50, box_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        bar_x, bar_y = box_x + 50, box_y + 100
        bar_width, bar_height = box_width - 100, 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (100, 200, 255), -1)
        
        progress_text = f"{int(progress * 100)}%"
        cv2.putText(frame, progress_text, (box_x + 150, box_y + 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        time_remaining = max(0, CALIBRATION_DURATION - (time.time() - self.calibration_start))
        time_text = f"{int(time_remaining)}s remaining"
        cv2.putText(frame, time_text, (box_x + 80, box_y + 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    def draw_main_ui(self, frame, concentration, avg_ear, face_tilt_x, face_tilt_y,
                     eyes_closed, session_time, w, h):
        """Draw main UI"""
        bar_height = 60
        cv2.rectangle(frame, (0, 0), (w, bar_height), (30, 30, 30), -1)
        
        concentration_text = f"{int(concentration)}%"
        if concentration >= 70:
            color = (0, 255, 100)
            status = "FOCUSED"
        elif concentration >= 50:
            color = (0, 200, 255)
            status = "MODERATE"
        else:
            color = (0, 100, 255)
            status = "DISTRACTED"
        
        cv2.putText(frame, concentration_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, status, (150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        bar_x, bar_y = 10, 50
        bar_width = w - 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 8), (50, 50, 50), -1)
        fill_width = int(bar_width * (concentration / 100))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 8), color, -1)
        
        time_text = f"Time: {int(session_time // 60):02d}:{int(session_time % 60):02d}"
        cv2.putText(frame, time_text, (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        if concentration < 50:
            flash_speed = 6
            alpha = 0.4 + 0.4 * np.sin(time.time() * flash_speed)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            warning_text = "PLEASE FOCUS!"
            warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            cv2.putText(frame, warning_text, ((w - warning_size[0]) // 2 + 2, h // 2 + 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
            cv2.putText(frame, warning_text, ((w - warning_size[0]) // 2, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    
    def reset_session(self, instance):
        """Reset tracking session"""
        self.calibrating = True
        self.calibration_start = time.time()
        self.concentration = 100.0
        self.ear_values = []
        self.face_orientations_x = []
        self.face_orientations_y = []
        self.alert_played = False
        self.ear_filter.reset()
        self.concentration_filter.reset()
        self.status_label.text = "Calibrating..."
    
    def toggle_pause(self, instance):
        """Toggle pause state"""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.text = "Resume"
            self.status_label.text = "PAUSED"
        else:
            self.pause_btn.text = "Pause"
    
    def on_stop(self):
        """Cleanup on app stop"""
        if self.cap is not None:
            self.cap.release()

if __name__ == '__main__':
    ConcentrationTrackerApp().run()

