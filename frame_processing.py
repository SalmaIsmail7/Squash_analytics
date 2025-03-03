import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F

# Load models
yolo_model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def forward(self, x):
        return self.encoder(x)


encoder = Autoencoder().encoder.to(device)
full_state_dict = torch.load("autoencoder.pth", map_location=device)
encoder_state_dict = {k.replace("encoder.", ""): v for k, v in full_state_dict.items() if k.startswith("encoder.")}
encoder.load_state_dict(encoder_state_dict)
encoder.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# Feature extraction
def extract_features(img):
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(img)
    return features.view(-1)


# Open video
video_path = "squash_match_2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

frame_counter = 0
class_names = ["Left Serve", "No Serve", "Right Serve"]
recent_predictions = []

# Player tracking and scoring
player_scores = {"Player A": 0, "Player B": 0}
player_embeddings = {"Player A": None, "Player B": None}
last_was_no_serve = True
players_assigned = False
update_interval = 150

# **NEW: Serve cooldown to prevent multiple points per serve**
serve_cooldown = 0  # Will count down to prevent double scoring


# Scoreboard display
def draw_scoreboard(frame):
    height, width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Player A: {player_scores['Player A']}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Player B: {player_scores['Player B']}", (width - 250, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % 5 != 0:
        continue

    results = yolo_model(frame)
    player_detections = []

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if conf > 0.7 and cls == 0:
                x1, y1, x2, y2 = map(int, box)
                player_detections.append((x1, y1, x2, y2))

    print(f"Detected Players: {len(player_detections)}")

    if not players_assigned and len(player_detections) >= 2:
        player_embeddings["Player A"] = extract_features(frame[player_detections[0][1]:player_detections[0][3],
                                                         player_detections[0][0]:player_detections[0][2]])
        player_embeddings["Player B"] = extract_features(frame[player_detections[1][1]:player_detections[1][3],
                                                         player_detections[1][0]:player_detections[1][2]])
        players_assigned = True
        print("✅ Players A and B embeddings assigned!")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = resnet_model(img)
        _, predicted = torch.max(outputs, 1)

    recent_predictions.append(class_names[predicted.item()])
    if len(recent_predictions) > 5:
        recent_predictions.pop(0)

    majority_vote = Counter(recent_predictions).most_common(1)[0][0] if len(
        recent_predictions) == 5 else "Processing..."

    # **Decrement serve cooldown**
    if serve_cooldown > 0:
        serve_cooldown -= 1

    if majority_vote in ["Left Serve", "Right Serve"] and last_was_no_serve and serve_cooldown == 0:
        if len(player_detections) >= 1 and players_assigned:
            current_embedding = extract_features(
                frame[player_detections[0][1]:player_detections[0][3], player_detections[0][0]:player_detections[0][2]]
            )

            similarity_A = torch.nn.functional.cosine_similarity(current_embedding, player_embeddings["Player A"],
                                                                 dim=0)
            similarity_B = torch.nn.functional.cosine_similarity(current_embedding, player_embeddings["Player B"],
                                                                 dim=0)

            if similarity_A > similarity_B:
                serving_player = "Player A"
            else:
                serving_player = "Player B"

            print(f"✅ Serve by: {serving_player} | Scored a point!")
            player_scores[serving_player] += 1

            # **Activate cooldown (prevent multiple scores in short time)**
            serve_cooldown = 30

        last_was_no_serve = False

    elif majority_vote == "No Serve":
        last_was_no_serve = True

    draw_scoreboard(frame)
    cv2.imshow("Squash Match Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
