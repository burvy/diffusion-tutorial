import matplotlib.pyplot as plt
import torch

from data import CHANNELS, IMAGE_SIZE
from diffusion import sample
from model import Unet
from visualize import to_image
import matplotlib.animation as animation

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
steps = [999, 800, 600, 400, 250, 150, 50, 0]
fig, axes = plt.subplots(1, len(steps), figsize=(2 * len(steps), 2.5))
for ax, t in zip(axes, steps):
    # frames[-1] is t=0, so frames[-1 - t] is timestep t, whatever TIMESTEPS is
    ax.imshow(to_image(frames[-1 - t][0]), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t={t}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("reverse_process.png", dpi=120)

print(f"saved samples.png and reverse_process.png")
print(f"final std {final.std():.3f}  (want near 1.0, not 4.8)")


fig, ax = plt.subplots(figsize=(3, 3))
ax.axis("off")
im = ax.imshow(to_image(frames[0][0]), cmap="gray", vmin=0, vmax=1, animated=True)

# every 5th frame keeps the file small; then hold on the result so it doesn't snap back
order = list(range(0, len(frames), 5)) + [len(frames) - 1] * 25

def update(i: int):
    im.set_data(to_image(frames[i][0]))
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=order, interval=40, blit=True)
ani.save("reverse.gif", writer="pillow", fps=25)
print("saved reverse.gif")
