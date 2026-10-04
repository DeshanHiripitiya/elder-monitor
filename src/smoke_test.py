import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8n-pose.pt")   # downloads automatically
cap = cv2.VideoCapture("data/raw/clip.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("data/processed/skeleton_out.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model(frame, verbose=False)[0]
    out.write(result.plot())      # draws boxes + skeleton

cap.release(); out.release()
print("done, fps =", fps)