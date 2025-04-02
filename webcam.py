import cv2
import numpy as np
from ultralytics import YOLO
from ocsort.ocsort import OCSort

# Inisialisasi model YOLO
model = YOLO("yolov8n.pt")  # Ganti dengan model yang Anda gunakan

# Inisialisasi OC-SORT tracker
tracker = OCSort(
    det_thresh=0.3,
    iou_threshold=0.3,
    use_byte=False
)

# Buka video
video_source = 0  # 0 untuk webcam, atau path ke file video
cap = cv2.VideoCapture(video_source)

# Warna untuk visualisasi
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi objek dengan YOLO
    results = model(frame)[0]
    
    # Ekstrak bounding boxes dan skor
    boxes = []
    scores = []
    
    for r in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, class_id = r
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
    
    # Format deteksi untuk OC-SORT
    detections = np.array(boxes) if len(boxes) > 0 else np.empty((0, 4))
    scores = np.array(scores) if len(scores) > 0 else np.empty(0)
    
    # Update tracker
    if len(detections) > 0:
        tracks = tracker.update(detections, scores)
    else:
        tracks = np.empty((0, 5))
    
    # Visualisasi hasil tracking
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        # Pilih warna berdasarkan track_id
        color = COLORS[track_id % len(COLORS)].tolist()
        
        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Gambar ID
        text = f"ID: {track_id}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Tampilkan frame
    cv2.imshow("OC-SORT Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
