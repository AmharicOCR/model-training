import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the trained model (replace with your model's file path)
model_path = '../models/best_model.pth'  # Update this with your actual model file path
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode
map_location=torch.device('cpu')

# Define the necessary transformation for the image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if necessary
    transforms.Resize((32, 32)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with the mean and std used during training
])

# Load the input image
image_path = '../image_dir/ገ.jpg'  # Update this with your actual image file path
image = Image.open(image_path)

# Apply transformations to the image
input_image = transform(image).unsqueeze(0)  # Add batch dimension

# Make the prediction
with torch.no_grad():  # No need to compute gradients during inference
    output = model(input_image)

# Print the raw model output
print("Raw model output:", output)

# If you want to inspect the raw logits or probabilities:
# If output is logits, it's common to apply softmax to get probabilities.
# If you want probabilities, uncomment the following lines:
softmax = torch.nn.Softmax(dim=1)
output_probabilities = softmax(output)
print("Raw probabilities:", output_probabilities)