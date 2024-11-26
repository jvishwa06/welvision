from ultralytics import YOLO
import snap7
from snap7.util import set_bool
import cv2
import time

model = YOLO("best.pt")  
model.to('cuda')
# Connect to PLC
plc = snap7.client.Client()
plc.connect("192.168.0.1", 0, 1)  # Replace with your PLC's IP, rack, slot

def read_proximity_status():
    """Reads the proximity sensor status from the PLC."""
    data = plc.read_area(snap7.types.Areas.MK, 0, 0, 1)  # Adjust area as needed
    return snap7.util.get_bool(data, 0, 0)

def trigger_slot_opening():
    """Sends a signal to the PLC to open the slot for defective rollers."""
    data = bytearray(1)
    set_bool(data, 0, 0, True)
    plc.write_area(snap7.types.Areas.MK, 0, 1, data)  # Adjust memory location as needed

# Camera setup
cap = cv2.VideoCapture(0) 

print("Camera is on and waiting for proximity detection...")

while True:
    try:
        if read_proximity_status():  # Check proximity sensor status
            ret, frame = cap.read()  
            if ret:
                print("Frame captured, analyzing...")
                results = model.predict(frame,device=0)
                defect_detected = False

                for result in results[0].boxes.data:
                    class_id = int(result[-1])  # Get class ID
                    class_name = model.names[class_id]  # Map ID to class name
                    if class_name == 'damage' or 'rust' or 'scratch':
                        defect_detected = True
                        print(f"Defect detected with class: {class_name}")
                        break

                if defect_detected:
                    trigger_slot_opening()
                else:
                    print("No defect detected.")
            time.sleep(0.1)  # Brief delay to avoid multiple triggers
    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
