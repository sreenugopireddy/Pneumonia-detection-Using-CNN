import sys, os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.data_loader import train_loader, val_loader
from models.mobilenet_model import build_model

device = torch.device("cpu")

model = build_model().to(device)

# ✅ class weighting (NORMAL gets higher weight)
class_weights = torch.tensor([1.6, 1.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# train classifier + last block
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=5e-4)

EPOCHS = 3

for epoch in range(EPOCHS):

    # -------- TRAIN --------
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # -------- VALIDATE --------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", running_loss/len(train_loader))
    print("Train Acc:", train_acc)
    print("Val Acc:", val_acc)

torch.save(model.state_dict(), "models/mobilenet_pneumonia.pth")
print("\nFine-tuned model saved.")
