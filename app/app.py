import sys, os
sys.path.append(os.path.abspath("."))

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.mobilenet_model import build_model
from rag.explain import build_prediction_explanation, rag_answer

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ---------------- CONFIG ----------------

st.set_page_config(layout="wide")
st.title("🫁 Pneumonia AI Assistant — CNN + PDF-RAG")

device = torch.device("cpu")
class_names = ["NORMAL", "PNEUMONIA"]


# ---------------- MODEL LOAD ----------------

@st.cache_resource
def load_model():
    m = build_model()
    m.load_state_dict(
        torch.load("models/mobilenet_pneumonia.pth", map_location=device)
    )
    m.eval()
    for p in m.parameters():
        p.requires_grad = True
    return m

model = load_model()
cam = GradCAM(model=model, target_layers=[model.features[-1][0]])


# ---------------- TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])


# ===================== UI =====================

left, center, right = st.columns(3)

uploaded = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg","png","jpeg"]
)


# ===================== ANALYSIS =====================

if uploaded:

    pil_img = Image.open(uploaded).convert("RGB")
    x = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]

    p_pneu = probs[1].item()

    if p_pneu > 0.80:
        pred = 1
    elif p_pneu < 0.45:
        pred = 0
    else:
        pred = 0

    label = class_names[pred]
    confidence = probs[pred].item()

    # -------- LEFT --------
    with left:
        st.subheader("X-ray")
        st.image(pil_img, use_column_width=True)

    # -------- CENTER --------
    with center:
        targets = [ClassifierOutputTarget(pred)]
        cam_map = cam(input_tensor=x, targets=targets)[0]
        rgb = np.array(pil_img.resize((224,224))) / 255.0
        overlay = show_cam_on_image(rgb, cam_map, use_rgb=True)

        st.subheader("Model Attention")
        st.image(overlay, use_column_width=True)

    # -------- RIGHT --------
    with right:
        st.subheader("Prediction")
        st.metric("Class", label)
        st.metric("Confidence", f"{confidence:.3f}")
        st.progress(float(confidence))

        if confidence > 0.85:
            st.success("High confidence")
        elif confidence > 0.60:
            st.warning("Moderate confidence")
        else:
            st.error("Low confidence")

        st.write({
            "NORMAL": float(probs[0]),
            "PNEUMONIA": float(probs[1])
        })

    # -------- RAG EXPLANATION (CORRECT PLACE) --------

    st.markdown("---")

    explanation = build_prediction_explanation(label, confidence)
    st.markdown(explanation)


# ===================== RAG CHATBOT =====================

st.sidebar.title("💬 PDF-RAG Medical Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

q = st.sidebar.text_input("Ask medical question")

if q:
    ans = rag_answer(q)
    st.session_state.chat.append((q, ans))

for q,a in st.session_state.chat:
    st.sidebar.write("**You:**", q)
    st.sidebar.write("**Bot:**", a)
