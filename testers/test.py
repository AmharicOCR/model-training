import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch import nn

# Step 1: Define your custom CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Adjust the number of input channels (1 for grayscale, 3 for RGB)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Change 3 to 1 if model was trained on grayscale
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adjust the size of the fully connected layer based on the output size of conv layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjust to match the saved model, check the size after pooling
        self.fc2 = nn.Linear(512, num_classes)  # Output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Step 2: Load the trained model
NUM_CLASSES = 317  # Adjust based on your dataset
model = CustomCNN(num_classes=NUM_CLASSES)

# Load the pre-trained weights
try:
    model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("This error might be caused by mismatched model architectures. Check input channels and layer sizes.")

model.to(device)
model.eval()

# Step 3: Preprocess images
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Adjust if your model trained on different dimensions
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if the model expects 1 channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust for grayscale normalization
])

# Step 4: Load and process images from a directory
def predict_images(image_dir, model, transform):
    class_names = os.listdir("../amharic_dataset")  # Assuming class names are directories in your dataset
    images = os.listdir(image_dir)
    predictions = {}

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")  # Read the image as RGB
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = class_names[predicted.item()]
                predictions[img_name] = predicted_label

            # Optional visualization
            plt.imshow(image)
            plt.title(f"Predicted: {predicted_label}")
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    return predictions

# Step 5: Predict and display results
image_dir = "../image_dir"  # Replace with the directory containing your test images
predictions = predict_images(image_dir, model, transform)

# Print all predictions
for img, pred in predictions.items():
    print(f"Image: {img}, Predicted Label: {pred}")
