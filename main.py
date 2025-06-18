import cv2
import numpy as np 
import time
from datetime import datetime
import os

# === Configuration ===
camera_index = 0  # Change if needed
output_dir = "/home/user/Desktop/CamFootage"  # Replace with your mounted drive path
min_area = 5000  # Minimum area size to detect motion (adjust based on dog size)
record_seconds_after_motion = 10  # How long to record after motion stops

# === Setup ===
cap = cv2.VideoCapture(camera_index)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame1 = cap.read()
ret, frame2 = cap.read()

motion_detected = False
out = None
last_motion_time = None

def get_output_filename():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(output_dir, f"dog_motion_{timestamp}.avi")

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    frame_motion = False
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        frame_motion = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Start or continue recording if motion is detected
    if frame_motion:
        if not motion_detected:
            print("Motion started")
            motion_detected = True
            filename = get_output_filename()
            out = cv2.VideoWriter(filename, fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))
        last_motion_time = time.time()

    if motion_detected:
        out.write(frame1)
        # Stop recording if time exceeded without motion
        if time.time() - last_motion_time > record_seconds_after_motion:
            print("Motion stopped, saving video.")
            motion_detected = False
            out.release()

    # Show preview (optional)
    cv2.imshow("Dog Motion Detection", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret or cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
if out and motion_detected:
    out.release()
cv2.destroyAllWindows()
