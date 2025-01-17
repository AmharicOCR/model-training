import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Step 1: Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AmharicCharacterModel(nn.Module):
    def init(self, num_classes):
        super(AmharicCharacterModel, self).init()
        self.base_model = models.resnet18(pretrained=True)  # Using ResNet18 as the base
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)  # Replace final layer

    def forward(self, x):
        return self.base_model(x)

# Load the model
NUM_CLASSES = 317
# len(os.listdir("amharic_dataset"))  # Automatically determine number of classes
model = AmharicCharacterModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("amharic_character_model.pth"))
model.to(device)
model.eval()

# Step 2: Preprocess the Custom Data
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Same size as used in training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your custom image
img_path = "amharic_dataset/ዶ/ዶ_120.png"  # Replace with the path to your image
image = Image.open(img_path).convert("RGB")

# Apply the transformation
image = transform(image)
image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

# Step 3: Make Prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

# Step 4: Convert prediction to label
# Class names come from folder names in alphabetical order
class_names = os.listdir("amharic_dataset")  # List of class names
predicted_label = class_names[predicted.item()]

print(f"Predicted label: {predicted_label}")

# Step 5: Optionally visualize the image and prediction
plt.imshow(Image.open(img_path))  # Display the original image
plt.title(f"Predicted label: {predicted_label}")
print(predicted_label)
plt.show()