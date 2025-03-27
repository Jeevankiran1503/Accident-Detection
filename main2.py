from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    results = model.train(data='AccidentsDetectionYOLOv8/data1.yaml', epochs=20, imgsz=640,batch=16)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
# import torch
# print(torch.cuda.is_available())  # Should return True
