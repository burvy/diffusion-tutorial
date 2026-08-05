import torch

TIMESTEPS = 300

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    add noise at each step
    """
    betas = linear_beta_schedule(TIMESTEPS)
    alphas = 1.0 - betas # signal surviving ONE step
    # ... all steps up to t
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return torch.linspace(0.0001, 0.02, timesteps)
