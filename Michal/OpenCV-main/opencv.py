import cv2
import numpy as np
from ultralytics import YOLO
import torch
import threading

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the camera
video_path = "rtsp://joebidensbalzak69:hijisdement420@192.168.0.101:554/stream1"  # Use 0 for the default camera #use: "rtsp://joebidensbalzak69:hijisdement420@192.168.0.101:554/stream1" voor de ip camera.
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening camera")
    exit()

# Get the framerate of the camera
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)  # Calculate the correct wait time for real-time playback

# Load the YOLOv8 model for human detection
model = YOLO('yolov8n.pt').to(device)  # Use yolov8n (nano) for speed; choose yolov8s or larger for more accuracy

# Background subtractor and optical flow settings
backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)  # Use KNN for faster background subtraction
ret, frame = cap.read()
original_size = frame.shape[1], frame.shape[0]  # Store the original frame size
downscale_size = (640, 360)  # Define the downscale size
prev_gray = cv2.cvtColor(cv2.resize(frame, downscale_size), cv2.COLOR_BGR2GRAY)

# Structuring element for noise reduction
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Cumulative mask for tracking detected humans
cumulative_mask = np.zeros_like(prev_gray, dtype=np.uint8)

# Shared variables for threading
frame_lock = threading.Lock()
frame = None
overlay_frame = None

def capture_frames():
    global frame
    while True:
        ret, new_frame = cap.read()
        if not ret:
            break
        with frame_lock:
            frame = new_frame

def process_frames():
    global frame, overlay_frame, prev_gray, cumulative_mask
    while True:
        with frame_lock:
            if frame is None:
                continue
            current_frame = frame.copy()

        # Downscale frame for faster processing
        small_frame = cv2.resize(current_frame, downscale_size)

        # Step 1: Background subtractor and masking moving objects
        fg_mask = backSub.apply(small_frame)
        _, thresh = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
        processed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Step 2: Optical Flow for accurate movement within human regions
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Update the previous frame for optical flow
        prev_gray = gray

        # Use the YOLOv8 model to detect humans
        results = model(small_frame)

        # Create a mask for human movements
        person_mask = np.zeros(processed_mask.shape, dtype=np.uint8)

        # Iterate through all detected objects
        for result in results:
            for box in result.boxes:
                if box.cls == 0 and box.conf > 0.3:  # Lower the confidence threshold to 30%
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates

                    # Create a mask within the detected area of a person
                    person_mask[y1:y2, x1:x2] = processed_mask[y1:y2, x1:x2]

                    # Update the cumulative mask
                    cumulative_mask[y1:y2, x1:x2] = 255

        # Blur the regions within the cumulative mask
        small_frame[cumulative_mask > 0] = cv2.GaussianBlur(small_frame[cumulative_mask > 0], (15, 15), 10)

        # Draw rectangles around the detected persons
        for result in results:
            for box in result.boxes:
                if box.cls == 0 and box.conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red with a thickness of 2

        # Create a temporary heatmap for only the current frame and masked motion areas
        heatmap_current = person_mask.astype(np.float32) + cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        heatmap_normalized = cv2.normalize(heatmap_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a color map (like in the example: blue-green-red)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Combine the heatmap with the original frame
        overlay_frame = cv2.addWeighted(small_frame, 0.6, heatmap_colored, 0.4, 0)

        # Upscale the processed frame back to the original size
        overlay_frame = cv2.resize(overlay_frame, original_size)

# Start the frame capture and processing threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
capture_thread.start()
process_thread.start()

while True:
    if overlay_frame is not None:
        # Show the result without details of moving objects outside the humans
        cv2.imshow("Heatmap with red rectangles around humans", overlay_frame)

    # Use the calculated wait time for real-time playback
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()