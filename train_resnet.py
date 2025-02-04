import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Add noise to use for augmentation
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# Define transformations for preprocessing images + augmentations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize images to 224x224
    transforms.RandomRotation(10),         # Randomly rotate images
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjusting brightness
    transforms.ToTensor(),                  # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    AddGaussianNoise(mean=0.0, std=0.05),
])

# Paths to the dataset folders
train_path = "./dataset/train"

# Load full dataset for training
train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)

# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print class-to-index mapping
print("Class-to-index mapping:", train_dataset.class_to_idx)

# Load pre-trained ResNet model and modify the last layer
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # Adjust for 3 classes
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for 10 epochs
epochs = 10
train_losses = []

print("Starting training...")

for epoch in range(epochs):
    # Training phase
    model.train()
    running_train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "serve_classifier_resnet50.pth")
print("Model saved as serve_classifier_resnet50.pth")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()




