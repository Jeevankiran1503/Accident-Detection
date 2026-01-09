from ultralytics import YOLO
import os

def train_yolo_model():
    # Initialize YOLOv8 model (using nano version for faster training)
    model = YOLO('yolov8n.pt')
    
    # Training parameters
    training_params = {
        'data': 'accident_detection.yaml',  # Path to your YAML file
        'epochs': 50,                     # Number of training epochs
        'imgsz': 640,                      # Input image size
        'batch': 16,                       # Batch size
        'device': 0,                       # GPU device (0) or CPU (-1)
        'patience': 50,                    # Early stopping patience
        'project': 'runs/train',           # Save directory
        'name': 'accident_detection',      # Experiment name
        'exist_ok': True,                  # Allow overwriting existing runs
        'optimizer': 'Adam',               # Optimizer
        'lr0': 0.001,                      # Initial learning rate
        'pretrained': True,                # Use pretrained weights
        'val': True                        # Validate during training
    }
    
    # Train the model
    model.train(**training_params)
    
    return model

def validate_model(model):
    # Validate the trained model
    metrics = model.val()
    print("Validation Metrics:")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50:95: {metrics.box.map:.4f}")
    
    return metrics

def setup_tracking(model):
    # Configure tracking parameters
    tracking_params = {
        'tracker': 'botsort.yaml',  # Use BoT-SORT tracker
        'conf': 0.3,               # Confidence threshold
        'iou': 0.5,                # IoU threshold for tracking
        'max_det': 100,            # Maximum number of detections per frame
        'track_high_thresh': 0.5,  # Threshold for high confidence tracks
        'track_low_thresh': 0.1,   # Threshold for low confidence tracks
        'new_track_thresh': 0.6,   # Threshold for new tracks
        'track_buffer': 30,        # Frames to keep lost tracks
        'match_thresh': 0.8        # Matching threshold
    }
    
    return tracking_params

def main():
    # Create YAML file for dataset configuration
    
    # Save YAML file
   
    # Train model
    print("Starting model training...")
    model = train_yolo_model()
    
    # Validate model
    print("\nValidating model...")
    metrics = validate_model(model)
    
    # Setup tracking parameters
    print("\nConfiguring tracking parameters...")
    tracking_params = setup_tracking(model)
    
    # Save the model
    model.save('accident_detection.pt')
    print("\nModel saved as 'accident_detection.pt'")
    
    # Export model to ONNX format for potential CNN integration
    model.export(format='onnx')
    print("Model exported to ONNX format")

if __name__ == '__main__':
    main()