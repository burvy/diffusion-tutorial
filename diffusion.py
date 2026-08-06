from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

TIMESTEPS = 300

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    add noise at each step
    """
    return torch.linspace(0.0001, 0.02, timesteps)

betas = linear_beta_schedule(TIMESTEPS)
alphas = 1.0 - betas # signal surviving ONE step
# ... all steps up to t
alphas_cumprod = torch.cumprod(alphas, dim=0)
# define alphas
alphas = 1. - betas
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(
    a: torch.Tensor,
    t: torch.Tensor,
    x_shape: torch.Size
) -> torch.Tensor:
    """
    Processes timestep tensors to be processable by tensor multiplication
    """
    batch_size = t.shape[0]
    # a.gather(-1, [12, 250, 3, 199]) -> [a[12], a[250], a[3], a[199]]
    output = a.to(t.device).gather(-1, t)
    # adds padding 1s to the end of the [4,] tensor to make it processable
    return output.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None
) -> torch.Tensor:
    """
    closed form derivation of the forward process for noising

    Note that running q_sample on the same image twice will yield a different
    result, since `torch.randn_like` draws a random number each time.
    This is important since one image produces infinite free training.

    If you want, you can fix the noise for testing using the noise parameter
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_ab = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_1mab = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_ab * x_start + sqrt_1mab * noise


def p_losses(
    denoise_model: nn.Module,
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
    loss_type: Literal["l1", "l2", "huber"] = "huber",
) -> torch.Tensor:
    """
    Loss in one training step,
    model predicts noise, compare against known answer
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    match loss_type:
        # same flat correction with any error size
        case "l1": # absolute value of error
            return F.l1_loss(noise, predicted_noise)
        # SCREAMS at big errors
        case "l2": # squares the error
            return F.mse_loss(noise, predicted_noise)
        # quieter as the error is smaller, one large error wont affect as much
        # as l2
        case "huber": # l2 if small, l1 if large
            return F.smooth_l1_loss(noise, predicted_noise)

@torch.no_grad() # nothing is training
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    t_index: int
) -> torch.Tensor:
    """
    Produce x_{t-1} from x_t
    Then add jitter/noise
    We have to add noise or else the sample becomes, say, the mean of
    all boots, instead of a specific boot looking thing. We dont want a
    blob that looks like the mean of all boots
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_ab_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # where x_{t-1} most likely sits according to the model
    # we dont subtract the whole predicted noise, rather
    # `betas_t` / sqrt_one_minus_ab_t of it
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_ab_t
    )

    if t_index == 0:
        return model_mean
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    # random noise is added back in
    return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
