import sys, os
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from training.data_loader import test_loader, class_names
from models.mobilenet_model import build_model

device = torch.device("cpu")

# -------- LOAD MODEL --------
model = build_model()
model.load_state_dict(
    torch.load("models/mobilenet_pneumonia.pth", map_location=device)
)
model.eval()

all_preds = []
all_labels = []

# -------- RUN TEST SET --------
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# -------- METRICS --------

cm = confusion_matrix(all_labels, all_preds)

print("\nClass names:", class_names)
print("\nConfusion Matrix:\n")
print(cm)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

acc = (all_preds == all_labels).mean()
print("\nTest Accuracy:", acc)
