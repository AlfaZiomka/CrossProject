import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the camera
video_path = 0  # Use 0 for the default camera
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
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=16)  # Lower varThreshold for higher sensitivity
ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Structuring element for noise reduction
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Cumulative mask for tracking detected humans
cumulative_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)

# Frame interval for processing
frame_interval = 19
frame_count = 0

# Function to process frames
def process_frame(frame, prev_gray, cumulative_mask):
    global model, backSub, kernel

    # Step 1: Background subtractor and masking moving objects
    fg_mask = backSub.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
    processed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

    # Step 2: Optical Flow for accurate movement within human regions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Update the previous frame for optical flow
    prev_gray = gray

    # Use the YOLOv8 model to detect humans
    results = model(frame)

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
    frame[cumulative_mask > 0] = cv2.GaussianBlur(frame[cumulative_mask > 0], (51, 51), 30)

    # Draw rectangles around the detected persons
    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.conf > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red with a thickness of 2

    # Create a temporary heatmap for only the current frame and masked motion areas
    heatmap_current = person_mask.astype(np.float32) + cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    heatmap_normalized = cv2.normalize(heatmap_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a color map (like in the example: blue-green-red)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Combine the heatmap with the original frame
    overlay_frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

    return overlay_frame, prev_gray, cumulative_mask

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_interval == 0:
        # Process the frame
        overlay_frame, prev_gray, cumulative_mask = process_frame(frame, prev_gray, cumulative_mask)

        # Show the result without details of moving objects outside the humans
        cv2.imshow("Heatmap with red rectangles around humans", overlay_frame)

    # Use the calculated wait time for real-time playback
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()