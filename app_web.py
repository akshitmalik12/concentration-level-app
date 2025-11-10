"""
Simple Web-Based Concentration Tracker
Uses browser camera (mobile-friendly)
"""
from flask import Flask, render_template_string, Response, jsonify, request
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

# Global processing state
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Tracking variables (shared state)
tracking_state = {
    'calibrating': True,
    'calibration_start': time.time(),
    'concentration': 100.0,
    'ear_baseline': None,
    'ear_std': None,
    'adaptive_threshold': EAR_DROP_THRESHOLD,
    'ear_values': [],
    'neutral_face_x': None,
    'neutral_face_y': None,
    'face_orientations_x': [],
    'face_orientations_y': [],
    'session_start_time': None,
    'eyes_closed_start': None,
    'ear_filter': SmoothingFilter(SMOOTHING_WINDOW),
    'concentration_filter': SmoothingFilter(SMOOTHING_WINDOW),
    'face_tilt_x_filter': SmoothingFilter(SMOOTHING_WINDOW),
    'face_tilt_y_filter': SmoothingFilter(SMOOTHING_WINDOW),
    'prev_time': time.time(),
}

def process_frame_data(frame_data):
    """Process frame data from browser camera"""
    try:
        # Decode base64 image
        if ',' in frame_data:
            image_data = base64.b64decode(frame_data.split(',')[1])
        else:
            image_data = base64.b64decode(frame_data)
        
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        h, w = frame.shape[:2]
        current_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        ts = tracking_state
        status = 'Calibrating...'
        session_time = 0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            if ts['calibrating']:
                left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
                right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2
                ts['ear_values'].append(avg_ear)
                
                face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
                ts['face_orientations_x'].append(face_tilt_x)
                ts['face_orientations_y'].append(face_tilt_y)
                
                elapsed = current_time - ts['calibration_start']
                if elapsed >= CALIBRATION_DURATION and len(ts['ear_values']) > 0:
                    ts['ear_baseline'] = np.mean(ts['ear_values'])
                    ts['ear_std'] = np.std(ts['ear_values'])
                    ts['adaptive_threshold'] = max(ts['ear_baseline'] * EAR_DROP_THRESHOLD, ts['ear_baseline'] - 2 * ts['ear_std'])
                    ts['neutral_face_x'] = np.mean(ts['face_orientations_x'])
                    ts['neutral_face_y'] = np.mean(ts['face_orientations_y'])
                    ts['calibrating'] = False
                    ts['session_start_time'] = current_time
                    ts['concentration'] = 100.0
                    status = 'FOCUSED'
            else:
                session_time = current_time - ts['session_start_time']
                
                left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
                right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2
                avg_ear = ts['ear_filter'].update(avg_ear)
                
                face_tilt_x, face_tilt_y = compute_face_orientation(landmarks, w, h)
                face_tilt_x = ts['face_tilt_x_filter'].update(face_tilt_x)
                face_tilt_y = ts['face_tilt_y_filter'].update(face_tilt_y)
                
                deviation_x = abs(face_tilt_x - ts['neutral_face_x'])
                deviation_y = abs(face_tilt_y - ts['neutral_face_y'])
                
                eyes_closed = avg_ear < ts['adaptive_threshold']
                
                dt = current_time - ts['prev_time']
                ts['prev_time'] = current_time
                
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
                    if ts['eyes_closed_start'] is None:
                        ts['eyes_closed_start'] = current_time
                    closed_duration = current_time - ts['eyes_closed_start']
                    if closed_duration > EYE_CLOSED_TIME_LIMIT:
                        penalty = min(50, closed_duration * 3 * 0.8)
                        eye_factor = max(50, 100 - penalty)
                else:
                    ts['eyes_closed_start'] = None
                    if orientation_factor >= 90:
                        eye_factor = min(100, ts['concentration'] + 1.5 * dt)
                    else:
                        eye_factor = 100.0
                
                ear_factor = 100.0
                if avg_ear < ts['ear_baseline']:
                    ear_ratio = avg_ear / (ts['ear_baseline'] + 1e-6)
                    if ear_ratio < 0.85:
                        penalty = (0.85 - ear_ratio) * 40
                        ear_factor = max(60, 100 - penalty)
                
                ts['concentration'] = 0.15 * eye_factor + 0.80 * orientation_factor + 0.05 * ear_factor
                ts['concentration'] = ts['concentration_filter'].update(ts['concentration'])
                ts['concentration'] = max(30, min(100, ts['concentration']))
                
                if ts['concentration'] >= 70:
                    status = "FOCUSED"
                elif ts['concentration'] >= 50:
                    status = "MODERATE"
                else:
                    status = "DISTRACTED"
        else:
            if not ts['calibrating']:
                ts['concentration'] = max(30, ts['concentration'] - 5)
                status = "NO FACE DETECTED"
        
        # Update state
        with state['lock']:
            state['concentration'] = ts['concentration']
            state['status'] = status
            state['calibrating'] = ts['calibrating']
            state['session_time'] = session_time
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()

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
            transform: scaleX(-1); /* Mirror effect */
        }
        #canvas {
            display: none;
        }
        #overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            padding: 15px;
            background: linear-gradient(to bottom, rgba(0,0,0,0.8), transparent);
            z-index: 10;
        }
        #concentration {
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        #status {
            font-size: 20px;
            margin-bottom: 10px;
        }
        #bar {
            width: 100%;
            height: 15px;
            background: #333;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 8px;
        }
        #bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #0f0, #ff0, #f00);
            transition: width 0.3s;
        }
        #time {
            font-size: 16px;
            opacity: 0.9;
        }
        .alert {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 48px;
            font-weight: bold;
            color: #f00;
            text-shadow: 0 0 20px #f00, 0 0 40px #f00;
            z-index: 20;
            animation: flash 0.5s infinite;
            pointer-events: none;
        }
        @keyframes flash {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .calibrating {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            z-index: 15;
        }
        .calibrating h2 {
            margin-bottom: 15px;
        }
        .calibrating-progress {
            width: 200px;
            height: 10px;
            background: #333;
            border-radius: 5px;
            margin: 15px auto;
            overflow: hidden;
        }
        .calibrating-progress-bar {
            height: 100%;
            background: #0af;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <div id="container">
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
        <div id="overlay">
            <div id="concentration">100%</div>
            <div id="status">Initializing...</div>
            <div id="bar">
                <div id="bar-fill" style="width: 100%"></div>
            </div>
            <div id="time">Time: 00:00</div>
        </div>
        <div id="calibrating-overlay" class="calibrating" style="display: none;">
            <h2>CALIBRATING...</h2>
            <div class="calibrating-progress">
                <div id="calibrating-progress-bar" class="calibrating-progress-bar" style="width: 0%"></div>
            </div>
            <div id="calibrating-time">8s remaining</div>
        </div>
        <div id="alert" style="display: none;" class="alert">PLEASE FOCUS!</div>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let stream = null;
        let sendingFrames = false;
        
        // Request camera access
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        facingMode: 'user',
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    } 
                });
                video.srcObject = stream;
                video.play();
                
                // Set canvas size
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    startSendingFrames();
                });
            } catch (err) {
                console.error('Error accessing camera:', err);
                document.getElementById('status').textContent = 'Error: Camera access denied';
            }
        }
        
        // Send frames to server for processing
        function startSendingFrames() {
            if (sendingFrames) return;
            sendingFrames = true;
            
            function sendFrame() {
                if (!stream || video.readyState !== video.HAVE_ENOUGH_DATA) {
                    requestAnimationFrame(sendFrame);
                    return;
                }
                
                // Draw video to canvas
                ctx.drawImage(video, 0, 0);
                
                // Convert to base64
                const imageData = canvas.toDataURL('image/jpeg', 0.7);
                
                // Send to server
                fetch('/api/process_frame', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ frame: imageData })
                }).catch(err => console.error('Error sending frame:', err));
                
                // Send next frame after delay (throttle to ~10 FPS for performance)
                setTimeout(sendFrame, 100);
            }
            
            sendFrame();
        }
        
        // Update UI from server
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
                    
                    // Show/hide alert
                    if (data.concentration < 50 && !data.calibrating) {
                        document.getElementById('alert').style.display = 'block';
                    } else {
                        document.getElementById('alert').style.display = 'none';
                    }
                    
                    // Show/hide calibrating overlay
                    if (data.calibrating) {
                        document.getElementById('calibrating-overlay').style.display = 'block';
                        const elapsed = Date.now() / 1000 - (data.session_time || 0);
                        const progress = Math.min(100, (elapsed / 8) * 100);
                        document.getElementById('calibrating-progress-bar').style.width = progress + '%';
                        const remaining = Math.max(0, 8 - elapsed);
                        document.getElementById('calibrating-time').textContent = Math.ceil(remaining) + 's remaining';
                    } else {
                        document.getElementById('calibrating-overlay').style.display = 'none';
                    }
                })
                .catch(err => console.error('Error updating UI:', err));
        }
        
        // Start camera and UI updates
        startCamera();
        setInterval(updateUI, 100);
        updateUI();
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    data = request.json
    if 'frame' in data:
        process_frame_data(data['frame'])
    return jsonify({'success': True})

@app.route('/api/status')
def api_status():
    with state['lock']:
        return jsonify({
            'concentration': state['concentration'],
            'status': state['status'],
            'calibrating': state['calibrating'],
            'session_time': state['session_time']
        })

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset tracking session"""
    with state['lock']:
        tracking_state['calibrating'] = True
        tracking_state['calibration_start'] = time.time()
        tracking_state['concentration'] = 100.0
        tracking_state['ear_values'] = []
        tracking_state['face_orientations_x'] = []
        tracking_state['face_orientations_y'] = []
        tracking_state['ear_filter'].values.clear()
        tracking_state['concentration_filter'].values.clear()
        state['concentration'] = 100.0
        state['status'] = 'Calibrating...'
        state['calibrating'] = True
        state['session_time'] = 0
    return jsonify({'success': True})

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
    print("Concentration Tracker - Web Version (Mobile Camera)")
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
