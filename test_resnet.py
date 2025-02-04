import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import os

# Set device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define transformations for the test dataset
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

# Path to the test dataset
test_path = "C:/Users/USER/PycharmProjects/Capstone/dataset/test"  # Update path

# Load the test dataset
test_dataset = datasets.ImageFolder(root=test_path, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Single image at a time

# Load the trained model
model = resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Adjust for 3 classes
model.load_state_dict(torch.load("serve_classifier_resnet50.pth"))  # Load trained weights
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Class-to-index mapping
class_names = test_dataset.classes
print("Class Names:", class_names)

# Accuracy counters
correct = 0
total = 0
class_correct = [0] * len(class_names)
class_total = [0] * len(class_names)

# Evaluate using 3 consecutive frames with Majority Voting
print("Starting evaluation with majority voting...")
frame_buffer = []  # Stores the last three frames

with torch.no_grad():
    for idx, (image, label) in enumerate(test_loader):
        image, label = image.to(device), label.to(device)

        # Store frames in a buffer
        frame_buffer.append(image)

        # Once we have 3 frames, classify using majority voting
        if len(frame_buffer) == 3:
            predictions = [torch.argmax(model(frame)) for frame in frame_buffer]  # Get class for each frame
            predicted_class = max(set(predictions), key=predictions.count)  # Majority voting

            # Update accuracy metrics
            total += 1
            class_total[label.item()] += 1
            if predicted_class == label.item():
                correct += 1
                class_correct[label.item()] += 1

            # Clear the frame buffer for the next set
            frame_buffer = []

# Calculate and print overall accuracy
overall_accuracy = 100 * correct / total
print(f"Overall Test Accuracy with Majority Voting: {overall_accuracy:.2f}%")

# Calculate per-class accuracy
for i in range(len(class_names)):
    if class_total[i] > 0:
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f"Accuracy for class {class_names[i]}: {class_accuracy:.2f}%")
    else:
        print(f"Accuracy for class {class_names[i]}: No samples in test set")

