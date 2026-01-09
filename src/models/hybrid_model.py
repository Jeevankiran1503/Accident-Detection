import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import numpy as np

# ------------------------------
# Thresholds
# ------------------------------
YOLO_THRESHOLD = 0.60   
RESNET_THRESHOLD = 0.70  

# ------------------------------
# Load YOLO Model
# ------------------------------
yolo_model = YOLO("accident_detection.pt")

# ------------------------------
# Load ResNet Model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 2)
)

resnet_model.load_state_dict(torch.load("best_resnet_accident_model.pth", map_location=device))
resnet_model.to(device)
resnet_model.eval()

class_names = ["Accident", "Non Accident"]

# ------------------------------
# Image Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Video Input
# ------------------------------
cap = cv2.VideoCapture("inference2.mp4")

if not cap.isOpened():
    print("Error: Video not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    accident_detected = False

    for r in results:
        for box in r.boxes:
            score = float(box.conf.cpu().numpy())
            if score < YOLO_THRESHOLD:
                continue 

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_tensor = transform(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

            # ResNet prediction
            with torch.no_grad():
                outputs = resnet_model(roi_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                confidence = conf.item()
                label = class_names[pred.item()]

            # Apply ResNet Threshold
            if label == "Accident" and confidence >= RESNET_THRESHOLD:
                accident_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Final Alert
    if accident_detected:
        cv2.putText(frame, "ACCIDENT ALERT!",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 0, 255), 4)

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
