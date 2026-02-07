import sys, os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.data_loader import train_loader, val_loader
from models.cnn_model import PneumoniaCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5

for epoch in range(EPOCHS):

    # ----- TRAIN -----
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # ----- VALIDATE -----
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
    print("Train Loss:", train_loss/len(train_loader))
    print("Train Acc:", train_acc)
    print("Val Acc:", val_acc)

# ---- save model ----
torch.save(model.state_dict(), "models/pneumonia_cnn.pth")
print("Model saved.")
