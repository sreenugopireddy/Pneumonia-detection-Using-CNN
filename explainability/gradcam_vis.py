import sys, os
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.mobilenet_model import build_model
from training.data_loader import class_names

device = torch.device("cpu")

# ---------------- LOAD MODEL ----------------

model = build_model()
model.load_state_dict(torch.load("models/mobilenet_pneumonia.pth", map_location=device))
model.eval()

# IMPORTANT — enable gradients
for p in model.parameters():
    p.requires_grad = True

# ✅ correct last conv layer for MobileNetV2
target_layers = [model.features[-1][0]]

cam = GradCAM(model=model, target_layers=target_layers)

# ---------------- IMAGE ----------------

IMG_PATH = "sample.jpeg"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

pil_img = Image.open(IMG_PATH).convert("RGB")
input_tensor = transform(pil_img).unsqueeze(0)

# ---------------- PREDICTION ----------------

output = model(input_tensor)
pred_class = output.argmax(1).item()

print("Prediction:", class_names[pred_class])

# ---------------- GRADCAM ----------------

targets = [ClassifierOutputTarget(pred_class)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

rgb_img = np.array(pil_img.resize((224,224))) / 255.0
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

cv2.imwrite("gradcam_output.jpg", cam_image)
print("Saved → gradcam_output.jpg")
