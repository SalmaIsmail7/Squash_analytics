import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO  # YOLOv8 for player detection
from PIL import Image  # For image processing

# Load YOLO model for player detection
yolo_model = YOLO("yolov8n.pt")  # Use a smaller model for speed

# Load the trained ResNet serve classification model
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet_model = resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 3)  # 3 classes: left serve, right serve, no serve
resnet_model.load_state_dict(torch.load("serve_classifier_resnet50.pth"))
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Define image preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match training size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Open the video file or live stream
video_path = "squash_match_1.mp4"  # Change to 0 for webcam or provide YouTube link with pafy
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

frame_counter = 0  # Counter to process every Nth frame

# Frame processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    frame_counter += 1
    if frame_counter % 5 != 0:  # Process every 5th frame for efficiency
        continue

    # YOLO Player Detection (only for visualization)
    results = yolo_model(frame)
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw player box

    # **Full Frame Classification Using ResNet**
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)  # Convert to PIL Image
    img = transform(img).unsqueeze(0).to(device)  # Apply transformations

    # Classify serve using full frame
    with torch.no_grad():
        outputs = resnet_model(img)
        _, predicted = torch.max(outputs, 1)

    # Get the predicted label
    class_names = ["Left Serve", "No Serve", "Right Serve"]
    serve_label = class_names[predicted.item()]
    cv2.putText(frame, f"Serve: {serve_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Squash Match Analysis", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
