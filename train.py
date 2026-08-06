import torch
from torch.optim import Adam

from data import CHANNELS, dataloader
from diffusion import TIMESTEPS, p_losses
from model import Unet

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
LEARNING_RATE = 1e-3

model = Unet(dim=64, dim_mults=(1, 2, 4), channels=CHANNELS).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE) # better than SGD

for epoch in range(EPOCHS):
    for step, (images, _) in enumerate(dataloader):
        optimizer.zero_grad()

        images = images.to(device)
        batch_size = images.shape[0]
        # unique timestep for each image
        t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()

        loss = p_losses(model, images, t, loss_type="huber")
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"epoch {epoch}  step {step:>4}  loss {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pt")
    print(f"epoch {epoch} done, saved model.pt")
