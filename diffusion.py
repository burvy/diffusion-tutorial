import torch

TIMESTEPS = 300

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    add noise at each step
    """
    return torch.linspace(0.0001, 0.02, timesteps)
