import matplotlib.pyplot as plt
import torch

from data import CHANNELS, IMAGE_SIZE
from diffusion import sample
from model import Unet
from visualize import to_image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(dim=64, dim_mults=(1, 2, 4), channels=CHANNELS).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()  # tells GroupNorm/dropout-style layers we are not training

frames = sample(model, image_size=IMAGE_SIZE, batch_size=8, channels=CHANNELS)

# the last frame, t=0 is the finished images
final = frames[-1]
fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
for ax, img in zip(axes, final):
    ax.imshow(to_image(img), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.savefig("samples.png", dpi=120)

# some images along the journey
steps = [299, 250, 200, 150, 100, 50, 25, 0]
fig, axes = plt.subplots(1, len(steps), figsize=(2 * len(steps), 2.5))
for ax, t in zip(axes, steps):
    ax.imshow(to_image(frames[299 - t][0]), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t={t}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("reverse_process.png", dpi=120)

print(f"saved samples.png and reverse_process.png")
print(f"final std {final.std():.3f}  (want near 1.0, not 4.8)")
