import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

# Step 1: Prepare and Explore the Dataset

class AmharicCharacterDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the character images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_labels = []
        self.img_paths = []
        self.label_to_idx = {}

        # Load all image paths and labels
        for idx, label in enumerate(os.listdir(data_dir)):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                self.label_to_idx[label] = idx  # Map label name to an index
                for img_name in os.listdir(label_dir):
                    if img_name.endswith(".png") or img_name.endswith(".jpg"):
                        img_path = os.path.join(label_dir, img_name)
                        self.img_paths.append(img_path)
                        self.img_labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]

        # Convert label (Amharic character) to numerical index
        label_idx = self.label_to_idx[label]

        # Open the image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_idx  # Return image and numerical label


# Data Transformation (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Dataset and DataLoader
data_dir = "amdtst"  # Provide the path to your dataset folder
dataset = AmharicCharacterDataset(data_dir=data_dir, transform=transform)

# Split the dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 2: Build the Model

class AmharicCharacterModel(nn.Module):
    def __init__(self, num_classes):
        super(AmharicCharacterModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # Using ResNet18 as the base
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)  # Replace final layer

    def forward(self, x):
        return self.base_model(x)


# Create the model with the number of classes equal to the number of unique characters
num_classes = len(os.listdir(data_dir))  # Number of unique character folders
model = AmharicCharacterModel(num_classes=num_classes)

# Step 3: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the Model
epochs = 1
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)  # labels are now numerical indices

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print batch loss (optional, can be verbose)
        if batch_idx % 10 == 0:  # Print every 10 batches (you can adjust this number)
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {running_loss/len(train_loader):.4f}")

    # Step 5: Validate the Model
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Step 6: Use OCR (Optional - Pytesseract for text extraction from images)
# Example of using pytesseract to extract text from an image

# img_path = "path_to_image"  # Provide path to an image you want to extract text from
# img = Image.open(img_path)
# text = pytesseract.image_to_string(img, lang='amh')  # 'amh' is the Amharic language code for Tesseract
#
# print(f"Extracted text from image: {text}")
#
# # Optionally, visualize an example image and its predicted label
# img, label = dataset[0]  # Take the first image from the dataset
# plt.imshow(img.permute(1, 2, 0))  # Convert from Tensor to HxWxC
# plt.title(f"Label: {label}")
# plt.show()
