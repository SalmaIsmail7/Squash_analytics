import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter

# Draw scoreboard at the top of the screen (Always Visible)
def draw_scoreboard(frame):
    height, width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)  # Black background for scoreboard

    # Player Scores
    cv2.putText(frame, f"Player 1: {player_scores['Player 1']}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Player 2: {player_scores['Player 2']}", (width - 250, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Serve Type Display (Centered)
    cv2.putText(frame, f"Serve: {majority_vote}", (width // 2 - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# Load YOLO Model for Player Detection
yolo_model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ResNet Serve Classifier
resnet_model = resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 3)
resnet_model.load_state_dict(torch.load("serve_classifier_resnet50.pth", map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

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

# Image transformation for autoencoder input
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Ensure consistency with autoencoder training
    transforms.ToTensor()
])

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Proper normalization
])

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
        detections = [list(map(int, box)) for r in results for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls) if conf > 0.7 and cls == 0]

        if len(detections) == 2:
            player_images["Player 1"].append(frame[detections[0][1]:detections[0][3], detections[0][0]:detections[0][2]])
            player_images["Player 2"].append(frame[detections[1][1]:detections[1][3], detections[1][0]:detections[1][2]])

        frame_count += 1

    cap.release()
    print("✅ Collected player data. Training autoencoder...")

    # Prepare dataset
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
        print(f"Epoch {epoch+1}/10 - Loss: {loss.item():.4f}")

    print("✅ Autoencoder trained on Player 1.")
    return autoencoder

# Start Pipeline
video_path = "squash_demo_5.mp4"
autoencoder = train_autoencoder(video_path)
cap = cv2.VideoCapture(video_path)

frame_counter = 0
serve_cooldown = 0
players_assigned = False  # Ensures we only assign players once

# Score tracking
player_scores = {"Player 1": 0, "Player 2": 0}

# Serve cooldown logic (prevents multiple scores per serve)
last_was_no_serve = True

# Serve classification variables
recent_predictions = []
class_names = ["Left Serve", "No Serve", "Right Serve"]


# Score tracking
player_scores = {"Player 1": 0, "Player 2": 0}
serve_cooldown = 0  # Prevent multiple scores per serve
recent_predictions = []
last_was_no_serve = True
last_serving_player = None  # Stores who served last
last_serve_position = None  # Stores whether they served from "Left" or "Right"
majority_vote = "Processing..."  # Ensure it's always defined

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % 5 != 0:
        continue  # Process every 5th frame for efficiency

    results = yolo_model(frame)
    player_detections = [list(map(int, box)) for r in results for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls) if conf > 0.7 and cls == 0]

    if len(player_detections) == 2:
        # Classify Players Using Autoencoder Reconstruction Error
        x1, y1, x2, y2 = player_detections[0]
        img1 = frame[y1:y2, x1:x2]

        x1, y1, x2, y2 = player_detections[1]
        img2 = frame[y1:y2, x1:x2]

        img1_tensor = transform(Image.fromarray(img1)).unsqueeze(0).to(device)
        img2_tensor = transform(Image.fromarray(img2)).unsqueeze(0).to(device)

        loss1 = torch.mean((autoencoder(img1_tensor) - img1_tensor) ** 2).item()
        loss2 = torch.mean((autoencoder(img2_tensor) - img2_tensor) ** 2).item()

        # Determine Player 1 and Player 2 using autoencoder classification
        if loss1 < loss2:
            player1_box, player2_box = player_detections[0], player_detections[1]
        else:
            player1_box, player2_box = player_detections[1], player_detections[0]

        x1_p1, y1_p1, x2_p1, y2_p1 = player1_box
        x1_p2, y1_p2, x2_p2, y2_p2 = player2_box

        # Determine who is physically on the left or right
        if x1_p1 < x1_p2:
            position_p1, position_p2 = "Left", "Right"
        else:
            position_p1, position_p2 = "Right", "Left"

        # Print position info for debugging
        print(f"📍 Player 1 ({position_p1}): x1={x1_p1}, y1={y1_p1}, x2={x2_p1}, y2={y2_p1}")
        print(f"📍 Player 2 ({position_p2}): x1={x1_p2}, y1={y1_p2}, x2={x2_p2}, y2={y2_p2}")

        # Serve Classification using ResNet
        img_for_pred = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_for_pred = Image.fromarray(img_for_pred)
        img_for_pred = resnet_transform(img_for_pred).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = resnet_model(img_for_pred)
            _, predicted = torch.max(outputs, 1)

        class_names = ["Left Serve", "No Serve", "Right Serve"]
        serve_prediction = class_names[predicted.item()]
        recent_predictions.append(serve_prediction)

        if len(recent_predictions) > 5:
            recent_predictions.pop(0)

        # Majority voting to smooth classification
        majority_vote = Counter(recent_predictions).most_common(1)[0][0] if len(recent_predictions) == 5 else "Processing..."
       #serve_type = Counter(recent_predictions).most_common(1)[0][0] if len( recent_predictions) == 5 else "Processing..."

        # **Fix: Reduce No Serve Bias by Only Voting on Recent Serves**
        if recent_predictions.count("Left Serve") > 1:
            majority_vote = "Left Serve"
        elif recent_predictions.count("Right Serve") > 1:
            majority_vote = "Right Serve"

        # Serve cooldown mechanism
        if majority_vote in ["Left Serve", "Right Serve"]:
            serve_cooldown = 30  # Reset cooldown only on valid serve
        else:
            serve_cooldown = max(0, serve_cooldown - 1)  # Gradually decrease otherwise

        # Award points based on serve classification
        if majority_vote in ["Left Serve", "Right Serve"] and last_was_no_serve:
            if majority_vote == "Left Serve" and position_p1 == "Left":
                serving_player = "Player 1"
            elif majority_vote == "Left Serve" and position_p2 == "Left":
                serving_player = "Player 2"
            elif majority_vote == "Right Serve" and position_p1 == "Right":
                serving_player = "Player 1"
            elif majority_vote == "Right Serve" and position_p2 == "Right":
                serving_player = "Player 2"
            else:
                serving_player = None  # Shouldn't happen, but just in case

            if serving_player:
                # Check if the same player served twice from the same position
                if serving_player == last_serving_player and position_p1 == last_serve_position:
                    print(f"❌ Invalid Serve! {serving_player} cannot serve twice in a row from the {position_p1} side.")
                else:
                    print(f"✅ {serving_player} scored a point from a {majority_vote}!")
                    player_scores[serving_player] += 1
                    serve_cooldown = 30  # Prevent multiple scores from the same serve

                # Update last serve info
                last_serving_player = serving_player
                last_serve_position = position_p1  # Store the position they served from

        # Update last serve state
        if majority_vote == "No Serve":
            last_was_no_serve = True
        else:
            last_was_no_serve = False

        # Draw bounding boxes with updated labels
        cv2.rectangle(frame, (x1_p1, y1_p1), (x2_p1, y2_p1), (255, 0, 0), 3)  # Blue for Player 1
        cv2.putText(frame, f"Player 1 ({position_p1})", (x1_p1, y1_p1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.rectangle(frame, (x1_p2, y1_p2), (x2_p2, y2_p2), (0, 255, 0), 3)  # Green for Player 2
        cv2.putText(frame, f"Player 2 ({position_p2})", (x1_p2, y1_p2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    draw_scoreboard(frame)  # Keep the scoreboard visible at all times

    # Show the frame
    cv2.imshow("Squash Match Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
