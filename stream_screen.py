import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time
import math
from flask import Flask, Response

# --- CONFIGURATION (High Sensitivity Mode) ---
MODEL_PATH = 'final_v2.pt'  
CONFIDENCE = 0.05           # EXTREMELY LOW: Catches faint/blurry arms
MOVEMENT_THRESHOLD = 5      # 5 Pixels: Catches tiny movements in background
TIME_TO_ALERT = 5.0         # Seconds before turning RED
HOST_IP = '0.0.0.0'         
PORT = 5000

# Stream Settings (Keeps phone video fast)
STREAM_WIDTH = 800          
JPEG_QUALITY = 70           

app = Flask(__name__)

print(f"Loading model: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Tracking Memory
last_move_time = {}
last_position = {}

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_frames():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        
        while True:
            # 1. Capture Frame
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            current_time = time.time()
            
            # 2. Run Tracking (HIGH RES MODE)
            # imgsz=1920: Forces AI to scan every pixel (slower, but accurate for small objects)
            # tracker="botsort.yaml": Better at "re-finding" objects that flicker
            results = model.track(frame, conf=CONFIDENCE, imgsz=1920, persist=True, tracker="botsort.yaml", verbose=False)
            
            # 3. Process IDs (Motion Logic)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Initialize
                    if track_id not in last_position:
                        last_position[track_id] = (cx, cy)
                        last_move_time[track_id] = current_time
                    
                    # Check Movement
                    dist = calculate_distance((cx, cy), last_position[track_id])
                    
                    if dist > MOVEMENT_THRESHOLD:
                        last_move_time[track_id] = current_time
                        last_position[track_id] = (cx, cy)
                        color = (0, 255, 0) # Green
                        status = "Active"
                    else:
                        time_still = current_time - last_move_time[track_id]
                        if time_still >= TIME_TO_ALERT:
                            color = (0, 0, 255) # RED
                            status = f"STOPPED ({int(time_still)}s)"
                        else:
                            color = (0, 255, 255) # Yellow
                            status = f"Idle ({int(time_still)}s)"
                    
                    # Draw Thicker Boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    
                    # Draw Label
                    label = f"ID {track_id}: {status}"
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            # 4. Fallback (If tracking glitches but detection works)
            elif len(results[0].boxes) > 0:
                 for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)

            # 5. RESIZE FOR PHONE (Critical for Speed)
            h, w = frame.shape[:2]
            new_h = int(h * (STREAM_WIDTH / w))
            frame_resized = cv2.resize(frame, (STREAM_WIDTH, new_h))

            # 6. Encode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ret, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- WEB PAGE ---
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Robotic Arm Monitor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #111; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 0;}
            h1 { color: #0f0; margin-top: 20px; font-size: 1.5rem; }
            .video-container { width: 100%; display: flex; justify-content: center; }
            img { width: 100%; max-width: 800px; height: auto; border-bottom: 2px solid #555; }
            .status { margin-top: 15px; color: #888; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <h1>🏭 Factory Live View</h1>
        <div class="video-container">
            <img src="/video_feed">
        </div>
        <p class="status">● High-Sensitivity Mode • 1920p Scanning</p>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"🚀 Streaming Started! Connect your phone to: http://<YOUR_IP>:5000")
    app.run(host=HOST_IP, port=PORT, debug=False)