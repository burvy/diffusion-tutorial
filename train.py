import torch
from torch.optim import Adam

from data import CHANNELS, dataloader
from diffusion import TIMESTEPS, p_losses
from model import Unet
from torch.optim.lr_scheduler import CosineAnnealingLR

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15
LEARNING_RATE = 1e-3

model = Unet(dim=64, dim_mults=(1, 2, 4), channels=CHANNELS).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE) # better than SGD
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    running = 0.0
    for step, (images, _) in enumerate(dataloader):
        optimizer.zero_grad()

        images = images.to(device)
        batch_size = images.shape[0]
        # unique timestep for each image
        t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()

        loss = p_losses(model, images, t, loss_type="huber")
        loss.backward()
        optimizer.step()

        running += loss.item()
        if step % 1 == 0:
            print(f"epoch {epoch}  step {step:>4}  loss {loss.item():.4f}")
        scheduler.step()
    print(f"epoch {epoch} done  mean loss {running / len(dataloader):.4f}, saved model.pt")
    torch.save(model.state_dict(), "model.pt")
    print(f"epoch {epoch} done, saved model.pt")
