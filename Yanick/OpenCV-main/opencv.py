import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialiseer de video
video_path = 'rtsp://Groepje6:bingchillin420@192.168.0.101:554/stream1'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Fout bij het openen van de video")
    exit()

# Haal de framerate van de video op
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)  # Bereken de juiste wachttijd voor real-time afspelen

# Laad het YOLOv8 model voor menselijke detectie
model = YOLO('yolov8s.pt').to(device)  # Gebruik yolov8n (nano) voor snelheid; kies yolov8s of groter voor meer nauwkeurigheid

# Achtergrondsubtractor en optical flow-instellingen
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=16)  # Verlaag varThreshold voor hogere gevoeligheid
ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Structurerend element voor ruisonderdrukking
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Cumulatief masker voor het bijhouden van gedetecteerde mensen
cumulative_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)

# Frame interval for processing
frame_interval = 10
frame_count = 0

# Function to process frames
def process_frame(frame, prev_gray, cumulative_mask):
    global model, backSub, kernel

    # Stap 1: Achtergrondsubtractor en maskeren van bewegende objecten
    fg_mask = backSub.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
    processed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

    # Stap 2: Optische Stroming voor nauwkeurige beweging binnen de mensen-regio's
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Update het vorige frame voor de optische stroom
    prev_gray = gray

    # Gebruik het YOLOv8 model om mensen te detecteren
    results = model(frame)

    # Maak een masker voor menselijke bewegingen
    person_mask = np.zeros(processed_mask.shape, dtype=np.uint8)

    # Doorloop alle gedetecteerde objecten
    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.conf > 0.3:  # Verlaag de zekerheid drempel naar 30%
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Verkrijg de bounding box-coördinaten

                # Creëer een masker binnen het gedetecteerde gebied van een persoon
                person_mask[y1:y2, x1:x2] = processed_mask[y1:y2, x1:x2]

                # Update het cumulatieve masker
                cumulative_mask[y1:y2, x1:x2] = 255

    # Blur de regio's binnen het cumulatieve masker
    frame[cumulative_mask > 0] = cv2.GaussianBlur(frame[cumulative_mask > 0], (51, 51), 30)

    # Teken rechthoeken om de gedetecteerde personen
    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.conf > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rood met een dikte van 2

    # Maak een tijdelijke heatmap voor alleen het huidige frame en gemaskeerde bewegingsgebieden
    heatmap_current = person_mask.astype(np.float32) + cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    heatmap_normalized = cv2.normalize(heatmap_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Pas een kleurenschema toe (zoals in het voorbeeld: blauw-groen-rood)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Combineer de heatmap met het originele frame
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

        # Toon het resultaat zonder details van bewegende objecten buiten de mensen
        cv2.imshow("Heatmap met rode rechthoeken om mensen", overlay_frame)

    # Gebruik de berekende wachttijd voor real-time afspelen
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()