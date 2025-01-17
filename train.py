import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.cuda.amp import autocast, GradScaler


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


# Data Transformation (resizing, augmentation, normalization)
transform = transforms.Compose([
    # transforms.RandomRotation(15),            # Random rotation by up to 15 degrees
    # transforms.RandomHorizontalFlip(),         # Random horizontal flip
    # transforms.RandomVerticalFlip(),           # Random vertical flip
    # transforms.RandomResizedCrop(32),         # Random crop and resize to 32x32
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Dataset and DataLoader
data_dir = "amharic_dataset"  # Provide the path to your dataset folder
dataset = AmharicCharacterDataset(data_dir=data_dir, transform=transform)

# Split the dataset into 85% training, 10% validation, and 5% testing sets
train_size = int(0.85 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Step 2: Define the Custom CNN Model with Batch Normalization

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # After applying 3 conv layers with 2x2 maxpooling, image size becomes (32 // 2^3) = 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjusted to match smaller feature map size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.pool(self.conv1(x))))  # Apply BN after conv1
        x = self.relu(self.bn2(self.pool(self.conv2(x))))  # Apply BN after conv2
        x = self.relu(self.bn3(self.pool(self.conv3(x))))  # Apply BN after conv3

        x = x.view(-1, 128 * 4 * 4)  # Flatten to match the fully connected layer input size
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Final output layer

        return x


# Create the model with the number of classes equal to the number of unique characters
num_classes = len(os.listdir(data_dir))  # Number of unique character folders
model = CustomCNN(num_classes=num_classes)

# Step 3: Define Loss Function, Optimizer and Scheduler
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Weight decay for regularization

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Mixed Precision Training (optional)
scaler = GradScaler()

# Step 4: Train the Model
best_accuracy = 0.0

epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Mixed Precision Training (optional)
        with autocast():  # Enable mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale the loss and backpropagate
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        # Print batch loss (optional, can be verbose)
        if batch_idx % 10 == 0:  # Print every 10 batches (you can adjust this number)
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {running_loss / len(train_loader):.4f}")

    # Step 5: Validate the Model
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect true labels and predictions for metrics calculation
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    val_accuracy = 100 * correct / total
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model")

    # Step 6: Adjust learning rate if needed
    scheduler.step(running_loss)  # Update scheduler based on the running loss

# Optional: Evaluate the best model on the test set
model.load_state_dict(torch.load('best_model.pth'))

model.eval()
test_correct = 0
test_total = 0
test_y_true = []
test_y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        # Collect true labels and predictions for metrics calculation
        test_y_true.extend(labels.numpy())
        test_y_pred.extend(predicted.numpy())

test_accuracy = 100 * test_correct / test_total
test_precision = precision_score(test_y_true, test_y_pred, average='macro')
test_recall = recall_score(test_y_true, test_y_pred, average='macro')
test_f1 = f1_score(test_y_true, test_y_pred, average='macro')

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1-score: {test_f1:.4f}")
