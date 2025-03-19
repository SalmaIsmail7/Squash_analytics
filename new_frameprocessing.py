import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def draw_scoreboard(frame, serve_type, player_scores):
    height, width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)  # Black rectangle for scoreboard
    cv2.putText(frame, f"Player 1: {player_scores['Player 1']}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Player 2: {player_scores['Player 2']}", (width - 250, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Serve Type: {serve_type}", (width // 2 - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

# Load YOLO Model for Player Detection
yolo_model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ResNet Serve Classifier
resnet_model = resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 3)
resnet_model.load_state_dict(torch.load("serve_classifier_resnet50.pth", map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Separate transform for ResNet classification
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Train Autoencoder on Player 1
def train_autoencoder(video_path):
    print("📹 Collecting player data for training autoencoder...")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    player_images = {"Player 1": [], "Player 2": []}

    while cap.isOpened() and frame_count < 100:  # Collect data for ~100 frames
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        detections = [list(map(int, box)) for r in results for box, conf, cls in
                      zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls) if conf > 0.7 and cls == 0]

        if len(detections) == 2:
            player_images["Player 1"].append(
                frame[detections[0][1]:detections[0][3], detections[0][0]:detections[0][2]])
            player_images["Player 2"].append(
                frame[detections[1][1]:detections[1][3], detections[1][0]:detections[1][2]])

        frame_count += 1

    cap.release()
    print("✅ Collected player data. Training autoencoder...")

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    player1_tensors = torch.stack([transform(Image.fromarray(img)) for img in player_images["Player 1"]]).to(device)

    # Train Autoencoder on Player 1
    autoencoder = Autoencoder().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):  # Small number of epochs for quick training
        optimizer.zero_grad()
        output = autoencoder(player1_tensors)
        loss = criterion(output, player1_tensors)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/10 - Loss: {loss.item():.4f}")

    print("✅ Autoencoder trained on Player 1.")
    return autoencoder


# Start Pipeline
video_path = "squash_trial_1.mp4"
autoencoder = train_autoencoder(video_path)
cap = cv2.VideoCapture(video_path)

frame_counter = 0
player_scores = {"Player 1": 0, "Player 2": 0}
recent_predictions = []
serve_cooldown = 0  # Prevents multiple scores per serve
last_was_no_serve = True  # Start assuming no serve happened
player1, player2 = "Player 1", "Player 2"  # Default assignment



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % 5 != 0:
        continue

    results = yolo_model(frame)
    player_detections = [list(map(int, box)) for r in results for box, conf, cls in
                         zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls) if conf > 0.7 and cls == 0]

    if len(player_detections) == 2:
        # Classify Players Using Autoencoder Reconstruction Error
        x1, y1, x2, y2 = player_detections[0]
        img1 = frame[y1:y2, x1:x2]

        x1, y1, x2, y2 = player_detections[1]
        img2 = frame[y1:y2, x1:x2]

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        img1_tensor = transform(Image.fromarray(img1)).unsqueeze(0).to(device)
        img2_tensor = transform(Image.fromarray(img2)).unsqueeze(0).to(device)

        loss1 = torch.mean((autoencoder(img1_tensor) - img1_tensor) ** 2).item()
        loss2 = torch.mean((autoencoder(img2_tensor) - img2_tensor) ** 2).item()

        if loss1 < loss2:
            player1, player2 = "Player 1", "Player 2"
        else:
            player1, player2 = "Player 2", "Player 1"

        # Draw bounding boxes
        for i, (x1, y1, x2, y2) in enumerate(player_detections):
            label = player1 if i == 0 else player2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert frame for Serve Prediction
    img_for_pred = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_for_pred = Image.fromarray(img_for_pred)
    img_for_pred = resnet_transform(img_for_pred).unsqueeze(0).to(device)

    # Predict Serve Type
    with torch.no_grad():
        outputs = resnet_model(img_for_pred)
        _, predicted = torch.max(outputs, 1)

    recent_predictions.append(["Left Serve", "No Serve", "Right Serve"][predicted.item()])
    if len(recent_predictions) > 5:
        recent_predictions.pop(0)

    # Determine Majority Serve Type
    serve_type = Counter(recent_predictions).most_common(1)[0][0] if len(recent_predictions) == 5 else "Processing..."

    # Serve Cooldown Mechanism
    if serve_cooldown > 0:
        serve_cooldown -= 1

    # Update Score Based on Serve
    if serve_type in ["Left Serve", "Right Serve"] and last_was_no_serve and serve_cooldown == 0:
        serving_player = player1 if serve_type == "Left Serve" else player2
        print(f"✅ Serve by: {serving_player} | Scored a point!")
        player_scores[serving_player] += 1
        serve_cooldown = 30  # Prevent multiple scores per serve
        last_was_no_serve = False  # Reset to prevent multiple scoring
    elif serve_type == "No Serve":
        last_was_no_serve = True  # Allow scoring when a serve actually happens

    draw_scoreboard(frame, serve_type, player_scores)

    cv2.imshow("Squash Match Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
