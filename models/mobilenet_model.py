import torch.nn as nn
from torchvision import models

def build_model():

    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    # Freeze all backbone first
    for param in model.features.parameters():
        param.requires_grad = False

    # ✅ Unfreeze LAST block for fine-tuning
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    return model
