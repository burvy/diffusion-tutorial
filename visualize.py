import matplotlib.pyplot as plt
import torch

from data import train_set
from diffusion import q_sample


def to_image(x: torch.Tensor) -> torch.Tensor:
    """(1, 28, 28) in [-1, 1] -> (28, 28) in [0, 1] for matplotlib"""
    return ((x + 1) / 2).squeeze().clamp(0, 1)


x_start = train_set[0][0]                    # one garment, already in [-1, 1]
steps = [0, 25, 50, 100, 150, 200, 250, 299]

fig, axes = plt.subplots(1, len(steps), figsize=(2 * len(steps), 2.5))
for ax, t in zip(axes, steps):
    noisy = q_sample(x_start.unsqueeze(0), torch.tensor([t]))
    ax.imshow(to_image(noisy), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t={t}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("forward_process.png", dpi=120)
print("saved forward_process.png")
