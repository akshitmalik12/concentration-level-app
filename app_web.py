"""
Simple Web-Based Concentration Tracker
Runs in browser - works on mobile without APK installation
"""
from flask import Flask, render_template_string, Response, jsonify
import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import threading
import base64

app = Flask(__name__)

# Constants
CALIBRATION_DURATION = 8
EYE_CLOSED_TIME_LIMIT = 2.5
FACE_TILT_THRESHOLD_X = 0.04
FACE_TILT_THRESHOLD_Y = 0.05
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

# Global state
state = {
    'concentration': 100.0,
    'status': 'Calibrating...',
    'calibrating': True,
    'session_time': 0,
    'frame': None,
    'lock': threading.Lock()
}

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

def process_frame():
    """Process video frames in background"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    calibrating = True
    calibration_start = time.time()
    concentration = 100.0
    ear_baseline = None
    ear_std = None
    adaptive_threshold = EAR_DROP_THRESHOLD
    ear_values = []
    neutral_face_x = None
    neutral_face_y = None
    face_orientations_x = []
    face_orientations_y = []
    session_start_time = None
    eyes_closed_start = None
    ear_filter = SmoothingFilter(SMOOTHING_WINDOW)
    concentration_filter = SmoothingFilter(SMOOTHING_WINDOW)
    face_tilt_x_filter = SmoothingFilter(SMOOTHING_WINDOW)
    face_tilt_y_filter = SmoothingFilter(SMOOTHING_WINDOW)
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w = frame.shape[:2]
        current_time = time.time()
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            if calibrating:
                left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
                right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2
                ear_values.append(avg_ear)
                
                face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
                face_orientations_x.append(face_tilt_x)
                face_orientations_y.append(face_tilt_y)
                
                elapsed = current_time - calibration_start
                if elapsed >= CALIBRATION_DURATION and len(ear_values) > 0:
                    ear_baseline = np.mean(ear_values)
                    ear_std = np.std(ear_values)
                    adaptive_threshold = max(ear_baseline * EAR_DROP_THRESHOLD, ear_baseline - 2 * ear_std)
                    neutral_face_x = np.mean(face_orientations_x)
                    neutral_face_y = np.mean(face_orientations_y)
                    calibrating = False
                    session_start_time = current_time
                    concentration = 100.0
            else:
                session_time = current_time - session_start_time
                
                left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
                right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2
                avg_ear = ear_filter.update(avg_ear)
                
                face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
                face_tilt_x = face_tilt_x_filter.update(face_tilt_x)
                face_tilt_y = face_tilt_y_filter.update(face_tilt_y)
                
                deviation_x = abs(face_tilt_x - neutral_face_x)
                deviation_y = abs(face_tilt_y - neutral_face_y)
                
                eyes_closed = avg_ear < adaptive_threshold
                
                dt = current_time - prev_time
                prev_time = current_time
                
                # Calculate concentration
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
                    if eyes_closed_start is None:
                        eyes_closed_start = current_time
                    closed_duration = current_time - eyes_closed_start
                    if closed_duration > EYE_CLOSED_TIME_LIMIT:
                        penalty = min(50, closed_duration * 3 * 0.8)
                        eye_factor = max(50, 100 - penalty)
                else:
                    eyes_closed_start = None
                    if orientation_factor >= 90:
                        eye_factor = min(100, concentration + 1.5 * dt)
                    else:
                        eye_factor = 100.0
                
                ear_factor = 100.0
                if avg_ear < ear_baseline:
                    ear_ratio = avg_ear / (ear_baseline + 1e-6)
                    if ear_ratio < 0.85:
                        penalty = (0.85 - ear_ratio) * 40
                        ear_factor = max(60, 100 - penalty)
                
                concentration = 0.15 * eye_factor + 0.80 * orientation_factor + 0.05 * ear_factor
                concentration = concentration_filter.update(concentration)
                concentration = max(30, min(100, concentration))
                
                # Draw UI
                bar_height = 60
                cv2.rectangle(frame, (0, 0), (w, bar_height), (30, 30, 30), -1)
                
                if concentration >= 70:
                    color = (0, 255, 100)
                    status = "FOCUSED"
                elif concentration >= 50:
                    color = (0, 200, 255)
                    status = "MODERATE"
                else:
                    color = (0, 100, 255)
                    status = "DISTRACTED"
                
                cv2.putText(frame, f"{int(concentration)}%", (10, 40),
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
        else:
            if not calibrating:
                concentration = max(30, concentration - 5)
                cv2.putText(frame, "NO FACE DETECTED", (w//2 - 150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update state
        with state['lock']:
            state['concentration'] = concentration
            state['status'] = 'Calibrating...' if calibrating else status
            state['calibrating'] = calibrating
            state['session_time'] = session_time if session_start_time else 0
            _, buffer = cv2.imencode('.jpg', frame)
            state['frame'] = base64.b64encode(buffer).decode('utf-8')
    
    cap.release()

# Start processing thread
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Concentration Tracker</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #000;
            color: #fff;
            overflow: hidden;
        }
        #container {
            position: relative;
            width: 100vw;
            height: 100vh;
        }
        #video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        #overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            padding: 20px;
            background: rgba(0,0,0,0.7);
            z-index: 10;
        }
        #concentration {
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        #status {
            font-size: 24px;
            margin-bottom: 10px;
        }
        #bar {
            width: 100%;
            height: 20px;
            background: #333;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        #bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #0f0, #ff0, #f00);
            transition: width 0.3s;
        }
        .alert {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 72px;
            font-weight: bold;
            color: #f00;
            text-shadow: 0 0 20px #f00;
            z-index: 20;
            animation: flash 0.5s infinite;
        }
        @keyframes flash {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
    </style>
</head>
<body>
    <div id="container">
        <img id="video" src="/video_feed" alt="Video Feed">
        <div id="overlay">
            <div id="concentration">100%</div>
            <div id="status">Calibrating...</div>
            <div id="bar">
                <div id="bar-fill" style="width: 100%"></div>
            </div>
            <div id="time">Time: 00:00</div>
        </div>
        <div id="alert" style="display: none;" class="alert">PLEASE FOCUS!</div>
    </div>
    <script>
        function updateUI() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('concentration').textContent = Math.round(data.concentration) + '%';
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('bar-fill').style.width = data.concentration + '%';
                    document.getElementById('time').textContent = 'Time: ' + 
                        Math.floor(data.session_time / 60).toString().padStart(2, '0') + ':' +
                        Math.floor(data.session_time % 60).toString().padStart(2, '0');
                    
                    if (data.concentration < 50) {
                        document.getElementById('alert').style.display = 'block';
                    } else {
                        document.getElementById('alert').style.display = 'none';
                    }
                });
        }
        setInterval(updateUI, 100);
        updateUI();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with state['lock']:
                frame_data = state['frame']
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       base64.b64decode(frame_data) + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with state['lock']:
        return jsonify({
            'concentration': state['concentration'],
            'status': state['status'],
            'calibrating': state['calibrating'],
            'session_time': state['session_time']
        })

if __name__ == '__main__':
    import socket
    
    # Find available port
    def find_free_port(start_port=5001, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None
    
    # Get IP address
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "YOUR_IP"
    
    port = find_free_port(5001)
    if port is None:
        print("❌ Error: Could not find an available port")
        exit(1)
    
    ip = get_local_ip()
    
    print("=" * 60)
    print("Concentration Tracker - Web Version")
    print("=" * 60)
    print(f"\n📱 To access on your phone:")
    print(f"1. Make sure your phone is on the same WiFi network")
    print(f"2. Open browser on phone and go to:")
    print(f"")
    print(f"   http://{ip}:{port}")
    print(f"")
    print(f"3. Allow camera access when prompted")
    print(f"\n🚀 Starting server on port {port}...")
    print("=" * 60)
    print(f"\nPress Ctrl+C to stop the server\n")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")

