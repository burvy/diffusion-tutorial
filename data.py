import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # PIL -> (1, 28, 28) float in [0, 1]
    transforms.Lambda(lambda t: (t * 2) - 1), # [0, 1] -> [-1, 1]
])

train_set = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

dataloader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
