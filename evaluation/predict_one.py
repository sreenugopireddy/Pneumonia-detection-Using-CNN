import sys, os
sys.path.append(os.path.abspath("."))
import torch
from PIL import Image
from torchvision import transforms

from models.cnn_model import PneumoniaCNN
from training.data_loader import train_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("models/pneumonia_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

img = Image.open("sample.jpeg").convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(x)
    pred = out.argmax(1).item()

print("Prediction:", train_dataset.classes[pred])
