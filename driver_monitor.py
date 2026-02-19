import cv2
import numpy as np
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully")
except ImportError:
    print("⚠️ MediaPipe not available - face detection disabled")
    MEDIAPIPE_AVAILABLE = False
import pygame
from ultralytics import YOLO
import time
import requests
import json
import os
import sys
import argparse
from supabase import create_client, Client as SupabaseClient
from twilio.rest import Client as TwilioClient
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_FROM = os.getenv("TWILIO_FROM")
ALERT_TO_PHONE = os.getenv("ALERT_TO_PHONE")
FAST2SMS_KEY = os.getenv("FAST2SMS_KEY")



def log_alert_to_supabase(alert_type, message):
    """Log alert to Supabase table"""
    try:
        print(f"🟢 log_alert_to_supabase CALLED: {alert_type} - {message}")
        data = {
            "created_at": datetime.utcnow().isoformat(),
            "type": alert_type,
            "message": message
        }
        response = supabase.table("driver_alerts").insert(data).execute()
        if response.data:
            print("✅ Supabase insert success:", response.data)
        else:
            print("⚠️ Supabase insert failed:", response.error)
    except Exception as e:
        print("❌ Error logging to Supabase:", e)



# -------- Parameters (tune these) ----------
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 15
MOUTH_AR_THRESHOLD = 0.6
MOUTH_AR_CONSEC_FRAMES = 12
YOLO_CONF = 0.45
YOLO_CONSEC_FRAMES = 10
ALERT_SOUND = "alert.mp3"
ALERT_MIN_INTERVAL_S = 2.0

pygame.mixer.init()
ALERT_CHANNEL = pygame.mixer.Channel(0)
ALERT_SOUND_OBJ = pygame.mixer.Sound(ALERT_SOUND)
_last_alert_ts = 0.0

def test_sms():
    """Test SMS functionality"""
    print("Testing SMS functionality...")
    print("⚠️ SMS functions are currently DISABLED for testing")
    test_message = "✅ Driver Monitor SMS Test - System is working correctly!"
    
    print("Testing Twilio SMS...")
    send_sms_twilio(test_message, ALERT_TO_PHONE)
    print("✅ Twilio SMS test completed (disabled)")
    
    print("Testing Fast2SMS...")
    send_sms_fast2sms(test_message, ALERT_TO_PHONE)
    print("✅ Fast2SMS test completed (disabled)")

# Non-blocking alert using pygame mixer
def play_alert():
    global _last_alert_ts
    now = time.time()
    if (now - _last_alert_ts) < ALERT_MIN_INTERVAL_S:
        return
    if not ALERT_CHANNEL.get_busy():
        ALERT_CHANNEL.play(ALERT_SOUND_OBJ)
        _last_alert_ts = now

# -------- SMS Sending Functions ------------
def send_sms_twilio(msg, to):
    #SMS DISABLED FOR TESTING - Uncomment to enable
    print(f"[SMS DISABLED] Would send Twilio SMS: {msg}")
    client = TwilioClient(TWILIO_SID, TWILIO_AUTH)
    message = client.messages.create(body=msg, from_=TWILIO_FROM, to=to)
    print("Twilio Message SID:", message.sid)

"""def send_sms_fast2sms(msg, to):
    #SMS DISABLED FOR TESTING - Uncomment to enable
    print(f"[SMS DISABLED] Would send Fast2SMS: {msg}")
    url = "https://www.fast2sms.com/dev/bulkV2"
    headers = {'authorization': FAST2SMS_KEY}
    payload = {
        'sender_id': 'TXTIND',
        'message': msg,
        'language': 'english',
        'route': 'v3',
        'numbers': to.replace("+91", "")
    }
    response = requests.post(url, data=payload, headers=headers)
    print("Fast2SMS Response:", response.json())"""

# Mediapipe init
if MEDIAPIPE_AVAILABLE:
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ MediaPipe face mesh initialized")
else:
    face_mesh = None

# YOLO init (pretrained COCO model) - optimized for speed
yolo = YOLO("yolov8n.pt")
# Optimize model for inference speed
yolo.overrides['conf'] = 0.3  # Set default confidence
yolo.overrides['verbose'] = False  # Disable verbose output

def lm_to_point(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.int32)

def compute_EAR(landmarks, eye_idx, w, h):
    eye_pts = np.array([lm_to_point(landmarks[i], w, h) for i in eye_idx])
    vertical_1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
    vertical_2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
    horizontal = np.linalg.norm(eye_pts[0] - eye_pts[3])
    if horizontal == 0:
        return 0.0
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def compute_MAR(landmarks, top_idx, bottom_idx, left_idx, right_idx, w, h):
    top = lm_to_point(landmarks[top_idx], w, h)
    bottom = lm_to_point(landmarks[bottom_idx], w, h)
    left = lm_to_point(landmarks[left_idx], w, h)
    right = lm_to_point(landmarks[right_idx], w, h)
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_IDX_TOP = 13
MOUTH_IDX_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

def open_camera():
    candidates = [  
        (1, cv2.CAP_DSHOW),
        (0, cv2.CAP_DSHOW),
        (1, cv2.CAP_MSMF),
        (0, cv2.CAP_MSMF),
        (2, cv2.CAP_DSHOW),
        (2, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
    ]
    for index, backend in candidates:
        cap_try = cv2.VideoCapture(index, backend)
        if cap_try.isOpened():
            # Optimize camera settings for speed
            cap_try.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap_try.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap_try.set(cv2.CAP_PROP_FPS, 30)            # Set FPS
            cap_try.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer for lower latency
            print(f"Camera opened: index={index}, backend={backend} (640x480 @ 30fps)")
            return cap_try
        else:
            cap_try.release()
    return None

# Argument parsing
def main():
    parser = argparse.ArgumentParser(description='Driver Monitor System')
    parser.add_argument('--test-sms', action='store_true', help='Test SMS functionality only')
    args = parser.parse_args()
    
    if args.test_sms:
        test_sms()
        return
    
    # Original driver monitoring code
    cap = open_camera()
    if cap is None:
        print("ERROR: Could not open any camera.")
        return
    
    print("Driver Monitor started. Press 'q' to quit.")
    if MEDIAPIPE_AVAILABLE:
        print("✅ Face detection (drowsiness/yawning) is ENABLED")
    else:
        print("⚠️ Face detection (drowsiness/yawning) is DISABLED - MediaPipe not available")
    print("✅ Object detection (distraction) is ENABLED")
    
    ear_counter = 0
    mar_counter = 0
    yolo_counter = 0
    frame_skip = 0  # Skip frames for better performance
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Skip frames for better performance (process every 2nd frame)
        frame_skip = (frame_skip + 1) % 2
        if frame_skip != 0:
            cv2.namedWindow("Driver Monitor", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Driver Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO object detection - optimized for speed
        yolo_results = yolo(frame, stream=True, verbose=False)
        distracted = False
        detected_this_frame = False

        for res in yolo_results:
            for box in res.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = yolo.model.names[cls]
                if conf > 0.3 and name in ["cell phone", "bottle", "cup", "banana", "apple", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "sandwich", "wine glass", "knife", "spoon", "toothbrush", "remote", "laptop", "book"]:
                    print(f"🚨 DISTRACTING OBJECT DETECTED: {name} (confidence: {conf:.2f})")
                    yolo_counter += 1
                    detected_this_frame = True
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    break
        if not detected_this_frame:
            yolo_counter = max(0, yolo_counter - 1)
        if yolo_counter >= YOLO_CONSEC_FRAMES:
            distracted = True

        # Face detection using MediaPipe (if available) - optimized
        drowsy, yawning = False, False
        
        if MEDIAPIPE_AVAILABLE and face_mesh is not None:
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                left_ear = compute_EAR(lm, LEFT_EYE, w, h)
                right_ear = compute_EAR(lm, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0
                mar = compute_MAR(lm, MOUTH_IDX_TOP, MOUTH_IDX_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT, w, h)

                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if ear < EYE_AR_THRESHOLD:
                    ear_counter += 1
                else:
                    ear_counter = max(0, ear_counter - 1)
                if ear_counter >= EYE_AR_CONSEC_FRAMES:
                    drowsy = True

                if mar > MOUTH_AR_THRESHOLD:
                    mar_counter += 1
                else:
                    mar_counter = max(0, mar_counter - 1)
                if mar_counter >= MOUTH_AR_CONSEC_FRAMES:
                    yawning = True
                
        else:
            # Show that face detection is disabled
            cv2.putText(frame, "Face Detection: Disabled (MediaPipe not available)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, "Install Python 3.12 to enable drowsiness/yawning detection", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Determine alert status and reason
        alert_reasons = []
        if distracted:
            alert_reasons.append("Distraction")
        if drowsy:
            alert_reasons.append("Drowsiness")
        if yawning:
            alert_reasons.append("Yawning")
        
        if alert_reasons:
            cv2.putText(frame, "ALERT! Driver at risk!", (50, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame, f"Reasons: {', '.join(alert_reasons)}", (50, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            play_alert()
            alert_message = f"Driver Alert: {', '.join(alert_reasons)} detected!"
            send_sms_twilio(alert_message, ALERT_TO_PHONE)
            #send_sms_fast2sms(alert_message, ALERT_TO_PHONE)
            log_alert_to_supabase(', '.join(alert_reasons), alert_message)
        else:
            cv2.putText(frame, "Status: OK", (50, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        pygame.mixer.quit()
    except Exception:
        pass

if __name__ == "__main__":main()
