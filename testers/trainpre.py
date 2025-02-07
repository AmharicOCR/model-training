import torch
from torchvision import transforms, models
from PIL import Image
import os

# Load the saved model
model_path = "../models/amharic_character_model.pth"  # Path to the saved model
model = models.resnet18(pretrained=False)

# Update the final fully connected layer to match your number of classes
num_classes = 34  # Update this to match the number of classes in your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the model's weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()  # Set the model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load label-to-index mapping (same as used during training)
label_to_idx = {  # Replace this with the mapping used during training
    "አ": 0,  # Example: አ (Amharic character) mapped to index 0
    "እ": 1,  # Example: እ mapped to index 1
    # Add other mappings here
}
idx_to_label = {v: k for k, v in label_to_idx.items()}  # Reverse mapping for predictions

# Function to predict the class of a single image
def predict_image(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)

    # Get the corresponding label
    predicted_label = idx_to_label[predicted_idx.item()]
    return predicted_label

# Test the model on a directory of test images
def test_model(test_dir):
    for img_name in os.listdir(test_dir):
        if img_name.endswith(".png") or img_name.endswith(".jpg"):
            img_path = os.path.join(test_dir, img_name)
            prediction = predict_image(img_path)
            print(f"Image: {img_name}, Predicted Label: {prediction}")

# Replace with your test images directory
test_dir = "../image_dir"
test_model(test_dir)
