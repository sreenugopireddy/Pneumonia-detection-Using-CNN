import sys, os
sys.path.append(os.path.abspath("."))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from models.mobilenet_model import build_model
from rag.explain import get_explanation

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ---------------- CONFIG ----------------

st.set_page_config(page_title="Pneumonia AI Detector", layout="centered")
st.title("🫁 Pneumonia X-Ray AI Analyzer")

device = torch.device("cpu")
class_names = ["NORMAL", "PNEUMONIA"]


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = build_model()
    model.load_state_dict(
        torch.load("models/mobilenet_pneumonia.pth", map_location=device)
    )
    model.eval()

    # GradCAM needs gradients enabled
    for p in model.parameters():
        p.requires_grad = True

    return model


model = load_model()

# correct target conv layer for MobileNetV2
target_layers = [model.features[-1][0]]
cam = GradCAM(model=model, target_layers=target_layers)


# ---------------- TRANSFORMS ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ---------------- UI ----------------

uploaded = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    x = transform(pil_img).unsqueeze(0)

    # ---------- PREDICTION ----------
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]

    # ✅ safer threshold to reduce false positives
    p_pneu = probs[1].item()

    if p_pneu > 0.80:
        pred = 1
    elif p_pneu < 0.45:
        pred = 0
    else:
        pred = 0   # treat uncertain as NORMAL to reduce false alarms


    label = class_names[pred]
    confidence = probs[pred].item()

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.3f}")

    st.write("Class probabilities:")
    st.write({
        "NORMAL": float(probs[0]),
        "PNEUMONIA": float(probs[1])
    })


    # ---------- GRAD-CAM ----------
    targets = [ClassifierOutputTarget(pred)]
    grayscale_cam = cam(input_tensor=x, targets=targets)[0]

    rgb_img = np.array(pil_img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.image(cam_image, caption="Model Attention (Grad-CAM)")


    # ---------- RAG EXPLANATION ----------
    explanation = get_explanation(label)

    st.subheader("AI Medical Explanation")
    st.write(explanation)
