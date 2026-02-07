import sys, os
sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt
from training.data_loader import train_loader


images, labels = next(iter(train_loader))

fig, axes = plt.subplots(2,4, figsize=(10,6))

for i, ax in enumerate(axes.flat):
    img = images[i].permute(1,2,0)
    img = (img * 0.5 + 0.5).numpy()
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(labels[i].item())
    ax.axis("off")

plt.show()
