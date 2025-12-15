import math
from typing import Optional, Tuple, Union

import torch
from diffusers.utils.torch_utils import randn_tensor


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flow_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: Optional[torch.Tensor],
    generator: Optional[torch.Generator] = None,
    return_stats: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    device = model_output.device

    sigma = sigmas[index].to(device)
    sigma_prev = sigmas[index + 1].to(device)
    sigma_max = sigmas[1].item()
    dt = sigma_prev - sigma  # negative

    pred_original_sample = latents - sigma * model_output

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta
    prev_sample_mean = (
        latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape, generator=generator, device=device, dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    noise_scale = std_dev_t * torch.sqrt(-1 * dt)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (noise_scale**2))
        - torch.log(noise_scale)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=device)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_stats:
        return prev_sample, pred_original_sample, log_prob, prev_sample_mean, noise_scale, dt
    return prev_sample, pred_original_sample, log_prob


def dance_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: Optional[torch.Tensor],
    grpo: bool,
    sde_solver: bool,
    return_stats: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(float(delta_t))

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if not grpo:
        return prev_sample_mean, pred_original_sample

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
        / (2 * (std_dev_t**2))
        - math.log(std_dev_t)
        - torch.log(
            torch.sqrt(2 * torch.as_tensor(math.pi, device=prev_sample_mean.device))
        )
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_stats:
        noise_scale = torch.as_tensor(std_dev_t, device=prev_sample_mean.device)
        return prev_sample, pred_original_sample, log_prob, prev_sample_mean, noise_scale, delta_t
    return prev_sample, pred_original_sample, log_prob
