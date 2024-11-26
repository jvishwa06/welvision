from ultralytics import YOLO
import cv2
import random
import time

# Load YOLOv8 model (default COCO model)
model = YOLO("yolov11.pt")  # Use 'yolov8n.pt', 'yolov8s.pt', etc., for pre-trained weights

def read_proximity_status():
    """Simulates the proximity sensor using a random trigger."""
    return True  # Randomly simulate proximity detection

def trigger_slot_opening():
    """Simulates the slot-opening mechanism."""
    print("Simulated: Slot opening signal sent.")

# Camera setup
cap = cv2.VideoCapture(0)  # Use webcam or test video

print("Camera is on and waiting for proximity detection...")

while True:
    try:
        if read_proximity_status():  # Simulate proximity sensor
            ret, frame = cap.read()  # Capture a frame
            if ret:
                print("Frame captured, analyzing...")
                results = model.predict(frame)
                person_detected = False

                for result in results[0].boxes.data:
                    class_id = int(result[-1])  # Get class ID
                    class_name = model.names[class_id]  # Map ID to class name
                    if class_name == 'person':  # Check for 'person'
                        person_detected = True
                        print(f"'Person' detected with class: {class_name}")
                        break

                if person_detected:
                    trigger_slot_opening()
                else:
                    print("No 'person' detected.")
            time.sleep(0.5)  # Adjust delay for testing
    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()