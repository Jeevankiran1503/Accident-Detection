from ultralytics import YOLO
import cv2
import os


model = YOLO("accident_detection.pt")


IGNORED_LABELS = ['no accident']
CONFIDENCE_THRESHOLD = 0.8

def infer_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)[0]  

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Filter by label and confidence
            if conf >= CONFIDENCE_THRESHOLD and label not in IGNORED_LABELS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Accident Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    video_path = "inference2.mp4"
    infer_on_video(video_path)
