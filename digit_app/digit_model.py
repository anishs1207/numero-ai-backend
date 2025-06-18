import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

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

model = CNN_DigitClassifier()
model.load_state_dict(torch.load("digit_model_cnn.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict_digit(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0] 
        predicted = torch.argmax(probs).item()
        confidence = probs[predicted].item()
        return predicted, confidence, probs.tolist()