# train_autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.nn.functional as F  # Import for interpolation


# **Custom Dataset Loader**
class AutoencoderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                            f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label


# **Set Dataset Path**
dataset_path = "autoencoder_training_data"

# **Define Transformations**
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure all images are 224x224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# **Load Dataset**
dataset = AutoencoderDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"✅ Loaded {len(dataset)} images for training!")


# **Improved Autoencoder Model**
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
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # **Force output to 224x224 using Interpolation (Fix Overshooting)**
        decoded = F.interpolate(decoded, size=(224, 224), mode="bilinear", align_corners=False)

        return decoded


# **Initialize Model**
device = "cuda" if torch.cuda.is_available() else "cpu"
autoencoder = Autoencoder().to(device)

# **Loss & Optimizer**
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)

# **Train the Autoencoder**
num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, _ in dataloader:
        images = images.to(device)

        # Forward pass
        outputs = autoencoder(images)

        # ✅ Ensure output shape matches input shape
        if outputs.shape != images.shape:
            print(f"❌ Shape Mismatch: Output {outputs.shape}, Expected {images.shape}")

        loss = criterion(outputs, images)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"🔄 Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# **Save the Full Autoencoder**
torch.save(autoencoder.state_dict(), "autoencoder.pth")
print("✅ Training complete! Full Autoencoder saved as 'autoencoder.pth'")

# **Plot Training Loss Curve**
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()
