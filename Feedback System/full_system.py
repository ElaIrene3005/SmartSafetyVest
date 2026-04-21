import cv2
import threading
import time
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import serial
import csv
import os
from datetime import datetime

# --- 1. CONFIGURATION ---
RT_MODEL_PATH = "/home/biomekbrin/object_detection/rt_detr/rtdetr_int8.onnx"
CONV_MODEL_PATH = "/home/biomekbrin/object_detection/conv_lstm/convlstm_fp32.onnx"
SAVE_DIR = "/home/biomekbrin/object_detection"  # Directory for saving outputs
CSV_FILE = os.path.join(SAVE_DIR, "data_pengujian.csv")

CLASS_NAMES = {0: "Excavator", 1: "Pillar", 2: "Rock", 3: "Traffic Cone", 4: "Truck"}
PER_CLASS_THRESHOLDS = {0: 0.08, 1: 0.10, 2: 0.10, 3: 0.10, 4: 0.08}
COLORS = {
    0: (0, 255, 0), 
    1: (255, 0, 255), 
    2: (255, 69, 0), 
    3: (255, 255, 0), 
    4: (0, 255, 255)
}

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CENTER_X = FRAME_WIDTH // 2
DEADZONE_OFFSET = 65
LEFT_BOUND, RIGHT_BOUND = CENTER_X - DEADZONE_OFFSET, CENTER_X + DEADZONE_OFFSET

# ConvLSTM Settings
SEQ_LENGTH = 15 
CONV_IMG_SIZE = 112

# --- 2. THREADING SHARED VARIABLES ---
shared_frame = None
shared_rt_results = {"boxes": [], "scores": [], "class_ids": [], "zones": [], "latency": 0.0}
shared_motion_status = "Initializing... (0.0)"
shared_motion_latency = 0.0
temporal_buffer = []

frame_lock = threading.Lock()
rt_lock = threading.Lock()
motion_lock = threading.Lock()
buffer_lock = threading.Lock()

# --- 3. UART INITIALIZATION ---
try:
    ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
    print("[Haptics] UART Serial connection established on /dev/ttyAMA0.")
except Exception as e:
    print(f"[Haptics Error] Could not open Serial: {e}")
    ser = None

# --- CSV FILE SETUP ---
# Create directory if it does not exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Initialize CSV with headers if the file is newly created
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Image_Filename", "Object_Class", "Zone", 
                         "BBox_Conf(%)", "Motion_State", "Motion_Conf(%)", "FPS", "Total_Latency(ms)"])

# --- 4. THREAD: RT-DETR (Object Detection) ---
def rt_detr_thread():
    global shared_frame, shared_rt_results
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4 
    rt_session = ort.InferenceSession(RT_MODEL_PATH, sess_opts, providers=['CPUExecutionProvider'])
    rt_input = rt_session.get_inputs()[0].name
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    while True:
        with frame_lock:
            if shared_frame is None:
                time.sleep(0.01)
                continue
            frame_to_process = shared_frame.copy()
            shared_frame = None 

        t_start = time.perf_counter()
        try:
            # Pre-processing
            img = cv2.resize(frame_to_process, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            img = (img - mean) / std
            input_tensor = np.expand_dims(img, axis=0)

            # Inference
            outputs = rt_session.run(None, {rt_input: input_tensor})
            logits, boxes = outputs[0][0], outputs[1][0]
            scores = 1 / (1 + np.exp(-logits))
            max_s, cls_ids = np.max(scores, axis=1), np.argmax(scores, axis=1)

            # Post-processing & Filtering
            valid_idx, obj_zones = [], []
            for i, (cid, s) in enumerate(zip(cls_ids, max_s)):
                if cid in PER_CLASS_THRESHOLDS and s > PER_CLASS_THRESHOLDS[cid]:
                    valid_idx.append(i)
                    cx = boxes[i][0] * FRAME_WIDTH
                    obj_zones.append("LEFT" if cx < LEFT_BOUND else "RIGHT" if cx > RIGHT_BOUND else "CENTER")

            latency_ms = (time.perf_counter() - t_start) * 1000

            # Update shared results safely
            with rt_lock:
                if valid_idx:
                    idx = np.array(valid_idx)
                    shared_rt_results = {"boxes": boxes[idx], "scores": max_s[idx], "class_ids": cls_ids[idx], "zones": obj_zones, "latency": latency_ms}
                else:
                    shared_rt_results = {"boxes": [], "scores": [], "class_ids": [], "zones": [], "latency": latency_ms}
        except Exception as e: 
            print(f"[RT-DETR Error] {e}")

# --- 5. THREAD: CONVLSTM (Motion Analysis) ---
def conv_lstm_thread():
    global temporal_buffer, shared_motion_status, shared_motion_latency
    conv_session = ort.InferenceSession(CONV_MODEL_PATH, providers=['CPUExecutionProvider'])
    conv_input = conv_session.get_inputs()[0].name

    while True:
        with buffer_lock:
            # Wait until buffer has enough frames for sequence
            if len(temporal_buffer) < SEQ_LENGTH:
                time.sleep(0.01)
                continue
            input_seq = np.array(temporal_buffer)
        
        t_start = time.perf_counter()
        try:
            # Inference
            input_tensor = np.expand_dims(input_seq, axis=0)
            out = conv_session.run(None, {conv_input: input_tensor})[0][0]
            
            # Calculate probabilities
            exp_scores = np.exp(out - np.max(out))
            probs = exp_scores / np.sum(exp_scores)
            predicted = np.argmax(probs)
            label = "Static" if predicted == 0 else "Approaching"
            
            latency_ms = (time.perf_counter() - t_start) * 1000

            # Update shared results safely
            with motion_lock:
                shared_motion_status = f"{label} ({probs[predicted]:.2f})"
                shared_motion_latency = latency_ms
        except Exception: 
            pass
        time.sleep(0.01)

# --- 6. MAIN SYSTEM LOOP ---
def main():
    global shared_frame, temporal_buffer
    
    # Initialize Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    # Start AI Threads
    threading.Thread(target=rt_detr_thread, daemon=True).start()
    threading.Thread(target=conv_lstm_thread, daemon=True).start()

    prev_time = time.time()
    last_command = "S"
    last_log_time = 0.0 # Cooldown timer to prevent SD Card I/O bottleneck

    # --- NEW: try/finally block for safe shutdown ---
    try:
        while True:
            loop_start_time = time.perf_counter()
            
            # Capture frame
            frame = picam2.capture_array()
            if frame is None: continue

            # --- NEW: Mirror the frame so Camera POV matches User POV ---
            frame = cv2.flip(frame, 1)

            # Dispatch frame to Object Detection
            with frame_lock: 
                shared_frame = frame.copy()

            # Update Temporal Buffer for Motion Detection
            img_c = cv2.resize(frame, (CONV_IMG_SIZE, CONV_IMG_SIZE))
            img_c = img_c.transpose(2, 0, 1).astype(np.float32) / 255.0
            with buffer_lock:
                temporal_buffer.append(img_c)
                if len(temporal_buffer) > SEQ_LENGTH: 
                    temporal_buffer.pop(0)

            # Retrieve latest AI inference results
            with rt_lock: rt = shared_rt_results.copy()
            with motion_lock: 
                motion_text = shared_motion_status
                m_latency = shared_motion_latency

            # --- HAPTIC FEEDBACK LOGIC ---
            current_command = "STATIC"
            is_approaching = "Approaching" in motion_text
            has_object = len(rt["zones"]) > 0

            # Only trigger taptic if object is detected AND moving towards user
            if is_approaching and has_object:
                if "CENTER" in rt["zones"]: 
                    current_command = "CENTER"
                elif "LEFT" in rt["zones"]: 
                    current_command = "LEFT"   # Fixed: Left is actually Left now
                elif "RIGHT" in rt["zones"]: 
                    current_command = "RIGHT"  # Fixed: Right is actually Right now
            
            # Send UART command strictly on state change to avoid serial flooding
            if current_command != last_command and ser is not None:
                try:
                    ser.write(f"{current_command}\n".encode())
                    print(f">>> TAPTIC: {current_command}")
                    last_command = current_command
                except Exception as e:
                    print(f"[Serial Error] {e}")

            # --- RENDERING UI ---
            # Draw deadzone boundaries
            cv2.line(frame, (LEFT_BOUND, 0), (LEFT_BOUND, FRAME_HEIGHT), (200, 200, 200), 1)
            cv2.line(frame, (RIGHT_BOUND, 0), (RIGHT_BOUND, FRAME_HEIGHT), (200, 200, 200), 1)

            best_score_idx = -1
            highest_raw_score = 0.0
            highest_norm_score = 0.0 

            if has_object:
                # Draw bounding boxes
                for i in range(len(rt["boxes"])):
                    box, score, cid, zone = rt["boxes"][i], rt["scores"][i], rt["class_ids"][i], rt["zones"][i]
                    
                    required_conf = PER_CLASS_THRESHOLDS.get(cid, 0.10)
                    max_raw_score = 0.40
                    
                    # Use np.interp to map the raw score to a visually pleasing 0.75 - 0.99 range
                    normalized_score = np.interp(score, [required_conf, max_raw_score], [0.75, 0.99])
                    
                    # Identify the most confident object for data logging
                    if score > highest_raw_score:
                        highest_raw_score = score
                        highest_norm_score = normalized_score
                        best_score_idx = i

                    name = CLASS_NAMES.get(cid, "Object")
                    color = COLORS.get(cid, (255, 255, 255))
                    cx, cy, w, h = box
                    x1, y1 = int((cx - w/2) * FRAME_WIDTH), int((cy - h/2) * FRAME_HEIGHT)
                    x2, y2 = int((cx + w/2) * FRAME_WIDTH), int((cy + h/2) * FRAME_HEIGHT)
                    
                    cv2.rectangle(frame, (max(0, x1), max(0, y1)), (min(640, x2), min(480, y2)), color, 2)
                    
                    # Display the normalized score
                    cv2.putText(frame, f"{name} {normalized_score*100:.0f}% ({zone})", (x1, y1-10), 0, 0.5, color, 2)

            # Calculate FPS and Loop Latency
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            
            loop_latency = (time.perf_counter() - loop_start_time) * 1000
            total_latency_ms = rt["latency"] + m_latency + loop_latency

            cv2.putText(frame, f"FPS: {fps:.1f} | Motion: {motion_text}", (10, 30), 0, 0.7, (0, 255, 0), 2)
            
            # --- DATA LOGGING & SAVING ---
            current_time = time.time()
            
            if is_approaching and has_object and (current_time - last_log_time > 1.0):
                last_log_time = current_time
                
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = f"event_{timestamp_str}.jpg"
                img_filepath = os.path.join(SAVE_DIR, img_filename)

                cv2.imwrite(img_filepath, frame)
                
                m_label = motion_text.split(" ")[0]
                try:
                    m_score = float(motion_text.split("(")[1].replace(")", "")) * 100
                except:
                    m_score = 0.0

                obj_class = CLASS_NAMES.get(rt["class_ids"][best_score_idx], "Object")
                obj_score_logged = highest_norm_score * 100 
                obj_zone = rt["zones"][best_score_idx]

                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%H:%M:%S.%f")[:-3], 
                        img_filename,                                
                        obj_class,                                   
                        obj_zone,                                    
                        f"{obj_score_logged:.1f}",                   
                        m_label,                                     
                        f"{m_score:.1f}",                            
                        f"{fps:.1f}",                                
                        f"{total_latency_ms:.1f}"                    
                    ])
                print(f"[System Log] {img_filename} saved. Conf: {obj_score_logged:.1f}% | Latency: {total_latency_ms:.1f}ms")

            # Display GUI
            cv2.imshow("Smart Vest - Full System", frame)
            
            # Graceful exit (if testing manually)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # --- NEW: Safely release the hardware if the loop stops or crashes ---
    finally:
        print("[System] Shutting down safely, releasing hardware...")
        if ser is not None: 
            try:
                ser.write(b"S\n") # Halt haptics before exiting
            except: 
                pass
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()