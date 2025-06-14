import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F # Import F for softmax, which is good practice

# Define the CNN model class (same as the training script, but essential to have here)
class CNN_DigitClassifier(nn.Module):
    def __init__(self):
        super(CNN_DigitClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Load CNN model
# Ensure 'digit_model_cnn.pth' is the name of your saved CNN model file
model = CNN_DigitClassifier()
model.load_state_dict(torch.load("digit_model_cnn.pth", map_location=torch.device("cpu")))
model.eval() # Set the model to evaluation mode

# Image preprocessing for CNN
# Note: The client-side handles the inversion (white on black) and initial 28x28 resize.
# This transform ensures it's a PyTorch Tensor and normalized.
transform = transforms.Compose([
    # transforms.Grayscale() is not strictly needed if the client sends a grayscale image already
    # transforms.Resize((28, 28)) is also handled client-side
    transforms.ToTensor(), # Converts PIL Image (or numpy array) to FloatTensor and scales pixel values to [0.0, 1.0]
    # MNIST typically has white digits on a black background, which means higher pixel values (closer to 1) for the digit itself.
    # No further normalization (like subtracting mean or dividing by std dev) is usually needed for basic MNIST.
])

def predict_digit(image: Image.Image):
    # Ensure the image is in the correct format (e.g., L mode for grayscale if coming from a diverse source)
    # The client-side processing should already produce a 28x28 grayscale image with white digit on black background.
    # transforms.ToTensor() will convert it to a FloatTensor, normalizing pixels to [0, 1].
    img_tensor = transform(image).unsqueeze(0) # Add a batch dimension (1, C, H, W) for the model

    with torch.no_grad(): # Disable gradient calculations for inference
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0] # Use F.softmax for clarity
        predicted = torch.argmax(probs).item()
        confidence = probs[predicted].item()
        # Return predicted digit, its confidence, and all class probabilities
        return predicted, confidence, probs.tolist()