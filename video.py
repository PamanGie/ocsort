import cv2
import numpy as np
from ultralytics import YOLO
from ocsort.ocsort import OCSort
import torch

# Inisialisasi model YOLO
model = YOLO("yolov8n.pt")

# Dapatkan nama kelas langsung dari model
class_names = model.names

# Inisialisasi OC-SORT tracker
tracker = OCSort(
    det_thresh=0.3,
    iou_threshold=0.3,
    use_byte=False
)

# Ganti dengan path video Anda
video_source = "oklusi.mp4"
cap = cv2.VideoCapture(video_source)

# Cek apakah video berhasil dibuka
if not cap.isOpened():
    print(f"Error: Tidak dapat membuka video {video_source}")
    exit()

# Warna untuk visualisasi
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    # Deteksi objek dengan YOLO
    results = model(frame)[0]
    
    # Ekstrak bounding boxes, skor, dan kelas
    detections = []
    
    for r in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, class_id = r
        # Format deteksi untuk OC-SORT: [x1, y1, x2, y2, score, class_id]
        detections.append([x1, y1, x2, y2, score, class_id])
    
    # Konversi ke torch tensor
    if len(detections) > 0:
        detections_tensor = torch.tensor(detections)
    else:
        detections_tensor = torch.zeros((0, 6))
    
    # Update tracker dengan deteksi
    tracks = tracker.update(detections_tensor, None)
    
    # Visualisasi hasil tracking
    for track in tracks:
        x1, y1, x2, y2, track_id, cls, conf = track  # [x1,y1,x2,y2,id,class,conf]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        class_id = int(cls)
        
        # Pilih warna berdasarkan track_id
        color = COLORS[track_id % len(COLORS)].tolist()
        
        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Dapatkan nama kelas dari model
        if class_id in class_names:
            class_name = class_names[class_id]
        else:
            class_name = f"Unknown-{class_id}"
        
        # Gambar ID dan nama kelas
        confidence = round(float(conf) * 100, 1)
        text = f"ID: {track_id} | {class_name} {confidence}%"
        
        # Tentukan posisi teks (sedikit di atas bounding box)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Tampilkan frame
    cv2.imshow("OC-SORT Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
