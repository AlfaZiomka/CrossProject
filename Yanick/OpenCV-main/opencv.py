import cv2
import numpy as np
from ultralytics import YOLO
import torch
import requests
import time
import matplotlib.pyplot as plt

# Function to send human count to the Flask endpoint
def send_human_count_to_db(count, durations):
    url = 'http://localhost:5000/update_count'  # Adjust the URL to match your Flask endpoint
    # Convert durations to a list of strings
    durations_str = [str(duration) for duration in durations]
    data = {'count': count, 'durations': durations_str}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Successfully sent human count: {count} and durations")
        else:
            print(f"Failed to send human count: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending human count: {e}")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the video
video_path = 'C:/Users/jhgh2/Documents/School/Project3/Yanick/OpenCV-main/bartest.mp4'  # '0' for webcam, 'video.mp4' for a video file, or 'rtsp://...' for an RTSP stream

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video")
    exit()

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)  # Calculate the appropriate wait time for real-time playback

# Load the YOLOv8 model for human detection
model = YOLO('yolov8s.pt').to(device)  # Use yolov8n (nano) for speed; choose yolov8s or larger for more accuracy

# Background subtractor and optical flow settings
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=16)  # Lower varThreshold for higher sensitivity
ret, frame = cap.read()
if not ret:
    print("Error reading the first frame")
    exit()

# Downscale the frame for faster processing
frame_small = cv2.resize(frame, (640, 360))
prev_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

# Structuring element for noise reduction
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Cumulative mask to track detected people
cumulative_mask = np.zeros_like(frame_small[:, :, 0], dtype=np.uint8)

# Frame interval for processing
frame_interval = 10
frame_count = 0

# List to store counts and timestamps
counts_and_timestamps = []

# Dictionary to track person IDs and their entry times
person_entry_times = {}

# Function to process frames
def process_frame(frame, prev_gray, cumulative_mask):
    global model, backSub, kernel, person_entry_times

    # Downscale the frame for faster processing
    frame_small = cv2.resize(frame, (640, 360))

    # Step 1: Background subtractor and mask moving objects
    fg_mask = backSub.apply(frame_small)
    _, thresh = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
    processed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

    # Step 2: Optical flow for accurate movement within the people regions
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Update the previous frame for optical flow
    prev_gray = gray

    # Use the YOLOv8 model to detect people
    results = model(frame_small)

    # Create a mask for human movements
    person_mask = np.zeros(processed_mask.shape, dtype=np.uint8)

    # Count the number of detected people
    user_count = 0  # Initialize user_count here

    # Iterate over all detected objects
    current_person_ids = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.conf > 0.3:  # Lower confidence threshold to 30%
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates

                # Create a mask within the detected area of a person
                person_mask[y1:y2, x1:x2] = processed_mask[y1:y2, x1:x2]

                # Update the cumulative mask
                cumulative_mask[y1:y2, x1:x2] = 255

                # Increment the user count
                user_count += 1

                # Assign a unique ID to each detected person
                person_id = id(box)
                current_person_ids.append(person_id)

                # If this is a new person, record their entry time
                if person_id not in person_entry_times:
                    person_entry_times[person_id] = time.time()

    # Remove people who are no longer detected
    for person_id in list(person_entry_times.keys()):
        if person_id not in current_person_ids:
            entry_time = person_entry_times.pop(person_id)
            duration = time.time() - entry_time
            print(f"Person {person_id} was in frame for {duration:.2f} seconds")

    # Log the count and timestamp
    counts_and_timestamps.append((time.time(), user_count))

    # Send the user count to the database
    durations = [time.time() - person_entry_times[person_id] for person_id in person_entry_times]
    send_human_count_to_db(user_count, durations)

    # Blur the regions within the cumulative mask
    frame_small[cumulative_mask > 0] = cv2.GaussianBlur(frame_small[cumulative_mask > 0], (51, 51), 30)

    # Draw rectangles and display duration for each person
    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.conf > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Assign a unique ID to each detected person
                person_id = id(box)

                # Calculate and display the duration
                if person_id in person_entry_times:
                    duration = time.time() - person_entry_times[person_id]
                    cv2.putText(frame_small, f"{duration:.2f}s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Create a temporary heatmap for the current frame and masked motion areas
    heatmap_current = person_mask.astype(np.float32) + cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    heatmap_normalized = cv2.normalize(heatmap_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a color map (such as in the example: blue-green-red)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Combine the heatmap with the original frame
    overlay_frame = cv2.addWeighted(frame_small, 0.6, heatmap_colored, 0.4, 0)

    # Resize overlay_frame back to the original frame size
    overlay_frame = cv2.resize(overlay_frame, (frame.shape[1], frame.shape[0]))

    return overlay_frame, prev_gray, cumulative_mask

def run_opencv():
    global cap, frame_count, frame_interval, prev_gray, cumulative_mask, person_entry_times

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_interval == 0:
            # Process the frame
            overlay_frame, prev_gray, cumulative_mask = process_frame(frame, prev_gray, cumulative_mask)

            # Show the result without details of moving objects outside the people
            cv2.imshow("Heatmap with red rectangles around people", overlay_frame)

        # Use the calculated wait time for real-time playback
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Generate the graph after the session ends
    timestamps, counts = zip(*counts_and_timestamps)
    timestamps = [time.strftime('%H:%M:%S', time.localtime(ts)) for ts in timestamps]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, counts, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Number of People')
    plt.title('Number of People Detected Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('people_count_over_time.png')
    plt.show()