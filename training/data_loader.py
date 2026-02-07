import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = "data"

IMG_SIZE = 224
BATCH_SIZE = 8   # CPU friendly

# -------------------------
# Transforms
# -------------------------

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -------------------------
# Full Train Dataset
# -------------------------

full_train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transforms
)

class_names = full_train_dataset.classes

# Split into train/val (80/20)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size]
)

# Test dataset (unchanged)
test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=val_test_transforms
)

# -------------------------
# Loaders
# -------------------------

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Debug Print
# -------------------------

print("Classes:", class_names)
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))
