# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import math
import os
from typing import Optional
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
import time
from torch.utils.data import DataLoader
import torch
import random

from torch.utils.data.distributed import DistributedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_wan21_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.config_io import dump_args_yaml
import cv2

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
from collections import deque
import numpy as np
from PIL import Image
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from fastvideo.utils.rollout_io import save_rollout_video
import json
from fastvideo.rewards.dispatcher import (
    compute_weighted_advantages,
    parse_reward_spec,
    RewardDispatcher,
)
from fastvideo.grpo.kl import disable_lora_adapters
from fastvideo.grpo.steps import dance_grpo_step, flow_grpo_step, sd3_time_shift
from fastvideo.grpo.ema import EMAModuleWrapper

    
def _reward_project_name(args, base):
    reward_names = "_".join(sorted(args.reward_weights.keys()))
    suffix = "_lora" if getattr(args, "use_lora", False) else ""
    return f"{base}_{reward_names}{suffix}"


def _parse_components(raw_components: Optional[str]) -> list[str]:
    if raw_components is None:
        return []
    if isinstance(raw_components, (list, tuple)):
        components = [str(item).strip() for item in raw_components]
    else:
        components = [item.strip() for item in str(raw_components).split(",")]
    return [component for component in components if component]


def _normalize_optional_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < 0:
        return None
    return float(value)


def _get_boundary_timestep(args) -> Optional[int]:
    boundary_ratio = _normalize_optional_float(getattr(args, "boundary_ratio", None))
    if boundary_ratio is None:
        return None
    num_train_timesteps = int(getattr(args, "num_train_timesteps", 1000))
    return int(boundary_ratio * num_train_timesteps)


def _get_cfg_scale(args, use_transformer_2: bool) -> float:
    if use_transformer_2:
        cfg_2 = _normalize_optional_float(getattr(args, "cfg_infer_2", None))
        if cfg_2 is not None:
            return cfg_2
    return float(args.cfg_infer)


def save_wan_lora_checkpoint(
    transformer,
    transformer_2,
    optimizer,
    pipeline,
    output_dir,
    step,
    epoch,
    rank,
    target_components,
):
    if rank > 0:
        return

    save_dir = os.path.join(output_dir, f"lora-checkpoint-{step}-{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    optim_path = os.path.join(save_dir, "lora_optimizer.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, optim_path)

    lora_layers = {}
    lora_metadata = {}
    lora_config = {"step": step, "lora_params": {}, "target_components": list(target_components)}

    if transformer is not None and "transformer" in target_components:
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        lora_layers["transformer"] = transformer_lora_layers
        lora_config["lora_params"]["transformer"] = {
            "lora_rank": transformer.config.lora_rank,
            "lora_alpha": transformer.config.lora_alpha,
            "target_modules": transformer.config.lora_target_modules,
        }

    if transformer_2 is not None and "transformer_2" in target_components:
        transformer_2_lora_layers = get_peft_model_state_dict(transformer_2)
        lora_layers["transformer_2"] = transformer_2_lora_layers
        lora_config["lora_params"]["transformer_2"] = {
            "lora_rank": transformer_2.config.lora_rank,
            "lora_alpha": transformer_2.config.lora_alpha,
            "target_modules": transformer_2.config.lora_target_modules,
        }

    if not lora_layers:
        raise ValueError("No LoRA layers found for saving. Check target_components.")

    pipeline._save_lora_weights(
        save_directory=save_dir,
        lora_layers=lora_layers,
        lora_metadata=lora_metadata,
        is_main_process=True,
    )
    config_path = os.path.join(save_dir, "lora_config.json")
    with open(config_path, "w") as f:
        json.dump(lora_config, f, indent=4)
    main_print(f"--> LoRA checkpoint saved at step {step}")


def load_wan_lora_weights(transformer, transformer_2, pipeline, checkpoint_dir):
    lora_state_dict = pipeline.lora_state_dict(checkpoint_dir)
    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer.")
    }
    transformer_2_state_dict = {
        f'{k.replace("transformer_2.", "")}': v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer_2.")
    }
    from diffusers.utils import convert_unet_state_dict_to_peft

    if transformer_state_dict and transformer is not None:
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer, transformer_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                main_print(
                    "Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f"{unexpected_keys}."
                )

    if transformer_2_state_dict:
        if transformer_2 is None:
            raise ValueError("Found transformer_2 LoRA weights but transformer_2 is not initialized.")
        transformer_2_state_dict = convert_unet_state_dict_to_peft(transformer_2_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_2, transformer_2_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                main_print(
                    "Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f"{unexpected_keys}."
                )


def resume_wan_lora_optimizer(optimizer, checkpoint_dir):
    optim_path = os.path.join(checkpoint_dir, "lora_optimizer.pt")
    if not os.path.exists(optim_path):
        return optimizer, 0

    optim_state = torch.load(optim_path, weights_only=False)
    step = optim_state.get("step", 0)
    state_dict = optim_state.get("optimizer", optim_state)
    optimizer.load_state_dict(state_dict)
    main_print(f"--> Successfully resumed LoRA optimizer from step {step}")
    return optimizer, step


def save_wan_full_checkpoint(
    transformer,
    transformer_2,
    output_dir,
    step,
    epoch,
    rank,
):
    if rank > 0:
        return
    save_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
    for name, model in (("transformer", transformer), ("transformer_2", transformer_2)):
        if model is None:
            continue
        model_dir = os.path.join(save_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        weight_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
        save_file(model.state_dict(), weight_path)
        config_dict = dict(model.config)
        if "dtype" in config_dict:
            del config_dict["dtype"]
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    main_print(f"--> checkpoint saved at step {step}")


def video_first_frame_to_pil(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return None

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频的第一帧")
        cap.release()
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(frame_rgb)

    cap.release()

    return pil_image


def load_eval_prompts(dataset, num_prompts, seed):
    if num_prompts <= 0:
        return []
    total = len(dataset)
    num_prompts = min(num_prompts, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), num_prompts)
    samples = []
    for idx in indices:
        data_item = dataset.data_anno[idx]
        prompt_embed_file = data_item["prompt_embed_path"]
        negative_prompt_embeds_file = data_item["negative_prompt_embeds_path"]
        prompt_embed = torch.load(
            os.path.join(dataset.prompt_embed_dir, prompt_embed_file),
            map_location="cpu",
            weights_only=True,
        )
        negative_prompt_embeds = torch.load(
            os.path.join(dataset.negative_prompt_embeds_dir, negative_prompt_embeds_file),
            map_location="cpu",
            weights_only=True,
        )
        caption = data_item.get("caption", "")
        samples.append((prompt_embed, negative_prompt_embeds, caption))
    return samples


def run_eval_videos(
    args,
    transformer,
    transformer_2,
    vae,
    eval_prompts,
    device,
    output_dir,
    step,
    rank,
    world_size,
):
    if not eval_prompts:
        return
    assigned_indices = [i for i in range(len(eval_prompts)) if i % world_size == rank]
    if not assigned_indices:
        return
    eval_root = os.path.join(output_dir, "eval_video", f"{step}_step")
    os.makedirs(eval_root, exist_ok=True)
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    video_processor = VideoProcessor(vae_scale_factor=8)
    was_training = transformer.training
    was_training_2 = transformer_2.training if transformer_2 is not None else None
    if args.init_same_noise:
        base_latents = torch.randn(
            (1, 16, ((args.t - 1) // 4) + 1, args.h // 8, args.w // 8),
            device=device,
            dtype=torch.bfloat16,
        )
    for idx in assigned_indices:
        prompt_embed, negative_prompt_embeds, caption = eval_prompts[idx]
        caption = str(caption)
        prompt_embed = prompt_embed.to(device)
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        if args.init_same_noise:
            input_latents = base_latents
        else:
            input_latents = torch.randn(
                (1, 16, ((args.t - 1) // 4) + 1, args.h // 8, args.w // 8),
                device=device,
                dtype=torch.bfloat16,
            )
        progress_bar = tqdm(
            range(0, args.sampling_steps),
            desc=f"Eval Sampling {idx}",
            disable=True,
        )
        with torch.no_grad():
            if getattr(args, "rationorm", False):
                z, latents, _, _, _ = run_sample_step(
                    args,
                    input_latents.clone(),
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    transformer_2,
                    prompt_embed.unsqueeze(0),
                    negative_prompt_embeds.unsqueeze(0),
                    True,
                )
            else:
                z, latents, _, _ = run_sample_step(
                    args,
                    input_latents.clone(),
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    transformer_2,
                    prompt_embed.unsqueeze(0),
                    negative_prompt_embeds.unsqueeze(0),
                    True,
                )
        vae.enable_tiling()
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
                    1, vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                video = vae.decode(latents, return_dict=False)[0]
                decoded_video = video_processor.postprocess_video(video)
        safe_caption = "".join(
            ch if ("0" <= ch <= "9" or "A" <= ch <= "Z" or "a" <= ch <= "z" or ch in "-_")
            else "_"
            for ch in caption
        )
        safe_caption = safe_caption.encode("ascii", errors="ignore").decode("ascii")
        safe_caption = safe_caption.strip("_")[:60] or "prompt"
        save_path = os.path.join(
            eval_root, f"sample_{idx:02d}_rank{rank}_{safe_caption}.mp4"
        )
        export_to_video(decoded_video[0], save_path, fps=16)
    if was_training:
        transformer.train()
    else:
        transformer.eval()
    if transformer_2 is not None:
        if was_training_2:
            transformer_2.train()
        else:
            transformer_2.eval()



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def _select_model_for_timestep(args, timestep_value, transformer, transformer_2):
    boundary_timestep = _get_boundary_timestep(args)
    if boundary_timestep is None or transformer_2 is None:
        return transformer, False
    use_transformer_2 = timestep_value < boundary_timestep
    return (transformer_2 if use_transformer_2 else transformer), use_transformer_2


def _forward_with_cfg(
    model,
    latents,
    timesteps,
    encoder_hidden_states,
    negative_prompt_embeds,
    cfg_scale,
):
    if cfg_scale > 1:
        with torch.autocast("cuda", torch.bfloat16):
            pred = model(
                hidden_states=torch.cat([latents, latents], dim=0),
                timestep=torch.cat([timesteps, timesteps], dim=0),
                encoder_hidden_states=torch.cat(
                    [encoder_hidden_states, negative_prompt_embeds], dim=0
                ),
                attention_kwargs=None,
                return_dict=False,
            )[0]
            model_pred, uncond_pred = pred.chunk(2)
            pred = uncond_pred.to(torch.float32) + cfg_scale * (
                model_pred.to(torch.float32) - uncond_pred.to(torch.float32)
            )
    else:
        with torch.autocast("cuda", torch.bfloat16):
            pred = model(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=None,
                return_dict=False,
            )[0]
        pred = pred.to(torch.float32)
    return pred

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        transformer_2,
        encoder_hidden_states, 
        negative_prompt_embeds, 
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        all_prev_sample_mean = [] if getattr(args, "rationorm", False) else None
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            current_model, use_transformer_2 = _select_model_for_timestep(
                args, timestep_value, transformer, transformer_2
            )
            current_model.eval()
            cfg_scale = _get_cfg_scale(args, use_transformer_2)
            pred = _forward_with_cfg(
                current_model,
                z,
                timesteps,
                encoder_hidden_states,
                negative_prompt_embeds,
                cfg_scale,
            )
            if args.grpo_step_mode == 'dance': 
                if getattr(args, "rationorm", False):
                    z, pred_original, log_prob, prev_sample_mean, _, _ = dance_grpo_step(
                        pred,
                        z.to(torch.float32),
                        args.eta,
                        sigmas=sigma_schedule,
                        index=i,
                        prev_sample=None,
                        grpo=True,
                        sde_solver=True,
                        return_stats=True,
                    )
                    all_prev_sample_mean.append(prev_sample_mean)
                else:
                    z, pred_original, log_prob = dance_grpo_step(
                        pred,
                        z.to(torch.float32),
                        args.eta,
                        sigmas=sigma_schedule,
                        index=i,
                        prev_sample=None,
                        grpo=True,
                        sde_solver=True,
                    )
            elif args.grpo_step_mode == 'flow': 
                if getattr(args, "rationorm", False):
                    z, pred_original, log_prob, prev_sample_mean, _, _ = flow_grpo_step(
                        model_output=pred,
                        latents=z.to(torch.float32),
                        eta=args.eta,
                        sigmas=sigma_schedule,
                        index=i,
                        prev_sample=None,
                        return_stats=True,
                    )
                    all_prev_sample_mean.append(prev_sample_mean)
                else:
                    z, pred_original, log_prob = flow_grpo_step(
                        model_output=pred,
                        latents=z.to(torch.float32),
                        eta=args.eta,
                        sigmas=sigma_schedule,
                        index=i,
                        prev_sample=None,
                    )
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        if getattr(args, "rationorm", False):
            all_prev_sample_mean = torch.stack(all_prev_sample_mean, dim=1)
            return z, latents, all_latents, all_log_probs, all_prev_sample_mean
        return z, latents, all_latents, all_log_probs

        
def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            negative_prompt_embeds, 
            transformer,
            transformer_2,
            timesteps,
            i,
            sigma_schedule,
            return_stats: bool = False,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    if transformer_2 is not None:
        transformer_2.train()

    boundary_timestep = _get_boundary_timestep(args)
    if boundary_timestep is None or transformer_2 is None:
        pred = _forward_with_cfg(
            transformer,
            latents,
            timesteps,
            encoder_hidden_states,
            negative_prompt_embeds,
            _get_cfg_scale(args, False),
        )
    else:
        use_transformer_2 = timesteps < boundary_timestep
        if torch.all(use_transformer_2):
            pred = _forward_with_cfg(
                transformer_2,
                latents,
                timesteps,
                encoder_hidden_states,
                negative_prompt_embeds,
                _get_cfg_scale(args, True),
            )
        elif torch.all(~use_transformer_2):
            pred = _forward_with_cfg(
                transformer,
                latents,
                timesteps,
                encoder_hidden_states,
                negative_prompt_embeds,
                _get_cfg_scale(args, False),
            )
        else:
            pred = torch.empty_like(latents, dtype=torch.float32)
            if torch.any(~use_transformer_2):
                pred[~use_transformer_2] = _forward_with_cfg(
                    transformer,
                    latents[~use_transformer_2],
                    timesteps[~use_transformer_2],
                    encoder_hidden_states[~use_transformer_2],
                    negative_prompt_embeds[~use_transformer_2],
                    _get_cfg_scale(args, False),
                )
            if torch.any(use_transformer_2):
                pred[use_transformer_2] = _forward_with_cfg(
                    transformer_2,
                    latents[use_transformer_2],
                    timesteps[use_transformer_2],
                    encoder_hidden_states[use_transformer_2],
                    negative_prompt_embeds[use_transformer_2],
                    _get_cfg_scale(args, True),
                )
            
    if args.grpo_step_mode == 'dance': 
        z, pred_original, log_prob, prev_sample_mean, noise_scale, dt = dance_grpo_step(
            pred,
            latents.to(torch.float32),
            args.eta,
            sigma_schedule,
            i,
            prev_sample=pre_latents.to(torch.float32),
            grpo=True,
            sde_solver=True,
            return_stats=True,
        )
    elif args.grpo_step_mode == 'flow': 
        z, pred_original, log_prob, prev_sample_mean, noise_scale, dt = flow_grpo_step(
            model_output=pred,
            latents=latents.to(torch.float32),
            eta=args.eta,
            sigmas=sigma_schedule,
            index=i,
            prev_sample=pre_latents.to(torch.float32),
            return_stats=True,
        )

    if return_stats:
        return log_prob, prev_sample_mean, noise_scale, dt
    return log_prob


def compute_kl_loss(prev_sample_mean, prev_sample_mean_ref, noise_scale_ref):
    diff_sq = (prev_sample_mean.to(torch.float32) - prev_sample_mean_ref.to(torch.float32)) ** 2
    reduce_dims = tuple(range(1, diff_sq.ndim))
    denom = 2.0 * (noise_scale_ref.to(torch.float32) ** 2)
    denom = torch.clamp(denom, min=1e-20)
    per_sample = diff_sq.mean(dim=reduce_dims) / denom
    return per_sample.mean()


class AdaptiveKLController:
    def __init__(
        self,
        init_beta: float,
        target: float,
        horizon: int,
        beta_min: float = 0.0,
        beta_max: float = 1e6,
        ema_alpha: float = 0.2,
    ):
        self.beta = float(init_beta)
        self.target = float(target)
        self.horizon = int(horizon)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.ema_alpha = float(ema_alpha)
        self._kl_ema = None

    def update(self, kl_value: float, n_steps: int = 1) -> float:
        kl_value = float(kl_value)
        if self._kl_ema is None:
            self._kl_ema = kl_value
        else:
            self._kl_ema = (1.0 - self.ema_alpha) * self._kl_ema + self.ema_alpha * kl_value

        if self.target <= 0:
            return self.beta

        horizon = max(self.horizon, 1)
        error = (self._kl_ema - self.target) / max(self.target, 1e-12)
        mult = math.exp(error * float(n_steps) / float(horizon))
        self.beta *= mult
        self.beta = float(max(self.beta_min, min(self.beta, self.beta_max)))
        return self.beta


def sample_reference_model(
    args,
    device, 
    transformer,
    transformer_2,
    vae,
    encoder_hidden_states, 
    negative_prompt_embeds, 
    reward_dispatcher,
    caption,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_prev_sample_mean = [] if getattr(args, "rationorm", False) else None
    reward_inputs = reward_dispatcher.build_reward_inputs()
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_negative_prompt_embeds = negative_prompt_embeds[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            if getattr(args, "rationorm", False):
                z, latents, batch_latents, batch_log_probs, batch_prev_sample_mean = run_sample_step(
                    args,
                    input_latents.clone(),
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    transformer_2,
                    batch_encoder_hidden_states,
                    batch_negative_prompt_embeds,
                    grpo_sample,
                )
                all_prev_sample_mean.append(batch_prev_sample_mean)
            else:
                z, latents, batch_latents, batch_log_probs = run_sample_step(
                    args,
                    input_latents.clone(),
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    transformer_2,
                    batch_encoder_hidden_states,
                    batch_negative_prompt_embeds, 
                    grpo_sample,
                )
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        vae.enable_tiling()
        
        video_processor = VideoProcessor(vae_scale_factor=8)
        rank = int(os.environ["RANK"])

        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                video = vae.decode(latents, return_dict=False)[0]
                decoded_video = video_processor.postprocess_video(video)
        save_path = save_rollout_video(
            decoded_video[0],
            args.output_dir,
            f"wan22_{rank}_{index}.mp4",
            fps=16,
        )

        for reward_name in reward_inputs:
            reward_inputs[reward_name].append(
                {"path": save_path, "prompt": batch_caption[0]}
            )
            
    reward_tensors, dim_reward = reward_dispatcher.compute_rewards(reward_inputs)
    
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    if getattr(args, "rationorm", False):
        all_prev_sample_mean = torch.cat(all_prev_sample_mean, dim=0)
    
    if getattr(args, "rationorm", False):
        return (
            reward_tensors,
            all_latents,
            all_log_probs,
            all_prev_sample_mean,
            sigma_schedule,
            dim_reward,
        )
    return reward_tensors, all_latents, all_log_probs, sigma_schedule, dim_reward


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    transformer,
    transformer_2,
    ref_transformer,
    ref_transformer_2,
    vae,
    reward_dispatcher,
    optimizer,
    lr_scheduler,
    encoder_hidden_states, 
    negative_prompt_embeds, 
    caption,
    noise_scheduler,
    max_grad_norm,
    kl_beta: Optional[float] = None,
    ema: Optional[EMAModuleWrapper] = None,
    ema_parameters: Optional[list[torch.nn.Parameter]] = None,
    ema_use_in_rollout: bool = False,
    ema_start_step: int = 0,
    global_step: int = 0,
):
    if kl_beta is None:
        kl_beta = float(getattr(args, "kl_beta", 0.0))
    kl_enabled = kl_beta > 0 or getattr(args, "kl_adaptive", False)
    total_loss = 0.0
    total_kl_loss = 0.0
    kl_loss_steps = 0
    optimizer.zero_grad()
    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        negative_prompt_embeds = repeat_tensor(negative_prompt_embeds)


        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    if getattr(args, "rationorm", False):
        ema_applied = False
        if (
            ema is not None
            and ema_parameters is not None
            and ema_use_in_rollout
            and global_step >= ema_start_step
        ):
            ema.copy_to(ema_parameters, store_temp=True)
            ema_applied = True
        (
            reward_tensors,
            all_latents,
            all_log_probs,
            all_prev_sample_mean,
            sigma_schedule,
            dim_reward,
        ) = sample_reference_model(
            args,
            device,
            transformer,
            transformer_2,
            vae,
            encoder_hidden_states,
            negative_prompt_embeds,
            reward_dispatcher,
            caption,
        )
        if ema_applied:
            ema.restore(ema_parameters)
    else:
        ema_applied = False
        if (
            ema is not None
            and ema_parameters is not None
            and ema_use_in_rollout
            and global_step >= ema_start_step
        ):
            ema.copy_to(ema_parameters, store_temp=True)
            ema_applied = True
        reward_tensors, all_latents, all_log_probs, sigma_schedule, dim_reward = sample_reference_model(
            args,
            device,
            transformer,
            transformer_2,
            vae,
            encoder_hidden_states,
            negative_prompt_embeds,
            reward_dispatcher,
            caption,
        )
        if ema_applied:
            ema.restore(ema_parameters)
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        **{
            f"reward_{name}": rewards.to(torch.float32)
            for name, rewards in reward_tensors.items()
        },
        "encoder_hidden_states": encoder_hidden_states,
        "negative_prompt_embeds": negative_prompt_embeds,
    }
    if getattr(args, "rationorm", False):
        samples["prev_sample_mean"] = all_prev_sample_mean[:, :-1]
    for name in args.reward_weights.keys():
        rewards = reward_tensors[name]
        gathered_reward = gather_tensor(rewards)
        if dist.get_rank() == 0:
            print(f"gathered_{name}_reward", gathered_reward)

    #计算advantage
    samples["advantages"], _ = compute_weighted_advantages(
        reward_tensors,
        args.reward_weights,
        gather_tensor=gather_tensor,
        use_group=args.use_group,
        num_generations=args.num_generations,
        apply_gdpo=getattr(args, "apply_gdpo", False),
    )

    
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    permute_keys = ["timesteps", "latents", "next_latents", "log_probs"]
    if getattr(args, "rationorm", False):
        permute_keys.append("prev_sample_mean")
    for key in permute_keys:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    for i,sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            need_stats = getattr(args, "rationorm", False) or kl_enabled
            if need_stats:
                new_log_probs, prev_sample_mean, noise_scale, dt = grpo_one_step(
                    args,
                    sample["latents"][:, _],
                    sample["next_latents"][:, _],
                    sample["encoder_hidden_states"],
                    sample["negative_prompt_embeds"],
                    transformer,
                    transformer_2,
                    sample["timesteps"][:, _],
                    perms[i][_],
                    sigma_schedule,
                    return_stats=True,
                )
            else:
                new_log_probs = grpo_one_step(
                    args,
                    sample["latents"][:, _],
                    sample["next_latents"][:, _],
                    sample["encoder_hidden_states"],
                    sample["negative_prompt_embeds"],
                    transformer,
                    transformer_2,
                    sample["timesteps"][:, _],
                    perms[i][_],
                    sigma_schedule,
                )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            if getattr(args, "rationorm", False):
                dt_f = dt.to(torch.float32)
                sqrt_dt = torch.sqrt(torch.clamp(torch.abs(dt_f), min=1e-20))
                sigma_t = (noise_scale.to(torch.float32) / sqrt_dt).mean()

                diff_sq = (prev_sample_mean.to(torch.float32) - sample["prev_sample_mean"][:, _].to(torch.float32)) ** 2
                reduce_dims = tuple(range(1, diff_sq.ndim))
                ratio_mean_bias = diff_sq.mean(dim=reduce_dims)

                scale = sqrt_dt.mean() * sigma_t
                scale = torch.clamp(scale, min=1e-20)
                ratio_mean_bias = ratio_mean_bias / (2.0 * (scale**2))
                ratio = torch.exp((new_log_probs - sample["log_probs"][:, _] + ratio_mean_bias) * scale)
            else:
                ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            if getattr(args, "rationorm", False):
                policy_loss = policy_loss / (sqrt_dt.mean() ** 2)
            loss = policy_loss

            kl_loss = None
            if kl_enabled:
                with torch.no_grad():
                    if ref_transformer is not None:
                        _, prev_sample_mean_ref, noise_scale_ref, dt_ref = grpo_one_step(
                            args,
                            sample["latents"][:, _],
                            sample["next_latents"][:, _],
                            sample["encoder_hidden_states"],
                            sample["negative_prompt_embeds"],
                            ref_transformer,
                            ref_transformer_2,
                            sample["timesteps"][:, _],
                            perms[i][_],
                            sigma_schedule,
                            return_stats=True,
                        )
                    else:
                        was_training = transformer.training
                        was_training_2 = transformer_2.training if transformer_2 is not None else None
                        transformer.eval()
                        if transformer_2 is not None:
                            transformer_2.eval()
                        try:
                            with disable_lora_adapters(transformer):
                                if transformer_2 is not None:
                                    with disable_lora_adapters(transformer_2):
                                        _, prev_sample_mean_ref, noise_scale_ref, dt_ref = grpo_one_step(
                                            args,
                                            sample["latents"][:, _],
                                            sample["next_latents"][:, _],
                                            sample["encoder_hidden_states"],
                                            sample["negative_prompt_embeds"],
                                            transformer,
                                            transformer_2,
                                            sample["timesteps"][:, _],
                                            perms[i][_],
                                            sigma_schedule,
                                            return_stats=True,
                                        )
                                else:
                                    _, prev_sample_mean_ref, noise_scale_ref, dt_ref = grpo_one_step(
                                        args,
                                        sample["latents"][:, _],
                                        sample["next_latents"][:, _],
                                        sample["encoder_hidden_states"],
                                        sample["negative_prompt_embeds"],
                                        transformer,
                                        None,
                                        sample["timesteps"][:, _],
                                        perms[i][_],
                                        sigma_schedule,
                                        return_stats=True,
                                    )
                        finally:
                            transformer.train(was_training)
                            if transformer_2 is not None:
                                transformer_2.train(was_training_2)

                kl_loss = compute_kl_loss(
                    prev_sample_mean, prev_sample_mean_ref, noise_scale_ref
                )
                if kl_beta > 0:
                    loss = loss + kl_beta * kl_loss

                kl_loss_to_log = kl_loss.detach()
                dist.all_reduce(kl_loss_to_log, op=dist.ReduceOp.SUM)
                kl_loss_to_log = kl_loss_to_log / dist.get_world_size()
                total_kl_loss += float(kl_loss_to_log.item())
                kl_loss_steps += 1

            loss = loss / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        if (i+1)%args.gradient_accumulation_steps==0:
            clip_params = list(p for p in transformer.parameters() if p.requires_grad)
            if transformer_2 is not None:
                clip_params.extend(p for p in transformer_2.parameters() if p.requires_grad)
            grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            for name in reward_tensors.keys():
                print(f"{name}_reward", sample[f"reward_{name}"].item())
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            if kl_enabled and kl_loss is not None:
                print("kl_loss", float(kl_loss.detach().item()))
                if getattr(args, "kl_adaptive", False):
                    print("kl_beta", float(kl_beta))
            print("final loss", loss.item())
        dist.barrier()
    mean_kl_loss = total_kl_loss / max(kl_loss_steps, 1)
    return total_loss, grad_norm.item(), dim_reward, mean_kl_loss


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    args.boundary_ratio = _normalize_optional_float(getattr(args, "boundary_ratio", None))
    args.cfg_infer_2 = _normalize_optional_float(getattr(args, "cfg_infer_2", None))
    target_components = _parse_components(getattr(args, "target_components", None))
    if not target_components:
        target_components = ["transformer", "transformer_2"]

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    reward_weights = parse_reward_spec(args.reward_spec)
    if not reward_weights:
        raise ValueError("No rewards configured; set --reward_spec.")
    args.reward_weights = reward_weights
    if rank <= 0:
        dump_args_yaml(args, args.output_dir)
    reward_dispatcher = RewardDispatcher(
        args=args,
        device=device,
        reward_weights=reward_weights,
        modality="video",
        clip_image_loader=video_first_frame_to_pil,
    )


    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    
    transformer = WanTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to(device)
    transformer_2 = WanTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
    ).to(device)

    if "transformer_2" in target_components and transformer_2 is None:
        raise ValueError("target_components includes transformer_2 but transformer_2 is not initialized.")

    pipe = None
    if getattr(args, "use_lora", False):
        from diffusers import WanPipeline

        pipe = WanPipeline
        transformer.requires_grad_(False)
        transformer_2.requires_grad_(False)
        target_modules = [
            "add_k_proj",
            "add_q_proj",
            "add_v_proj",
            "to_add_out",
            "to_k",
            "to_out.0",
            "to_q",
            "to_v",
        ]
        if "transformer" in target_components:
            transformer_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                init_lora_weights=True,
                target_modules=target_modules,
            )
            transformer.add_adapter(transformer_lora_config)
            transformer.config.lora_rank = args.lora_rank
            transformer.config.lora_alpha = args.lora_alpha
            transformer.config.lora_target_modules = target_modules

        if "transformer_2" in target_components:
            transformer_2_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                init_lora_weights=True,
                target_modules=target_modules,
            )
            transformer_2.add_adapter(transformer_2_lora_config)
            transformer_2.config.lora_rank = args.lora_rank
            transformer_2.config.lora_alpha = args.lora_alpha
            transformer_2.config.lora_target_modules = target_modules

        if args.resume_from_lora_checkpoint:
            load_wan_lora_weights(transformer, transformer_2, pipe, args.resume_from_lora_checkpoint)
    else:
        transformer.requires_grad_("transformer" in target_components)
        if transformer_2 is not None:
            transformer_2.requires_grad_("transformer_2" in target_components)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, (WanTransformerBlock), args.selective_checkpointing
        )
        if transformer_2 is not None:
            apply_fsdp_checkpointing(
                transformer_2, (WanTransformerBlock), args.selective_checkpointing
            )
    

    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model (optional, for KL regularization)
    ref_transformer = None
    ref_transformer_2 = None
    kl_should_load_ref = getattr(args, "kl_beta", 0.0) > 0 or getattr(args, "kl_adaptive", False)
    if kl_should_load_ref:
        if args.kl_reference_model_name_or_path:
            ref_transformer = WanTransformer3DModel.from_pretrained(
                args.kl_reference_model_name_or_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            ).to(device)
            ref_transformer_2 = WanTransformer3DModel.from_pretrained(
                args.kl_reference_model_name_or_path,
                subfolder="transformer_2",
                torch_dtype=torch.bfloat16,
            ).to(device)
            ref_transformer.requires_grad_(False)
            ref_transformer.eval()
            ref_transformer_2.requires_grad_(False)
            ref_transformer_2.eval()
        else:
            assert getattr(args, "use_lora", False), (
                "KL regularization requires either a separate ref_transformer "
                "(set --kl_reference_model_name_or_path) or a model that supports adapter disabling "
                "(enable --use_lora)."
            )
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()
    if transformer_2 is not None:
        transformer_2.train()

    noise_scheduler = None

    params_to_optimize = list(p for p in transformer.parameters() if p.requires_grad)
    if transformer_2 is not None:
        params_to_optimize.extend(p for p in transformer_2.parameters() if p.requires_grad)
    ema = None
    if getattr(args, "use_ema", False):
        ema = EMAModuleWrapper(
            params_to_optimize,
            decay=args.ema_decay,
            update_step_interval=args.ema_update_interval,
            device=device,
        )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    if getattr(args, "use_lora", False) and args.resume_from_lora_checkpoint:
        optimizer, init_steps = resume_wan_lora_optimizer(
            optimizer, args.resume_from_lora_checkpoint
        )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    eval_prompts = None
    if (
        args.eval_every_steps > 0
        and args.output_dir is not None
    ):
        seed = args.seed if args.seed is not None else 0
        eval_prompts = load_eval_prompts(
            train_dataset,
            args.eval_num_prompts,
            seed,
        )
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    
    total_samples = len(train_dataloader)
    effective_batch_size = args.train_sp_batch_size * args.sp_size
    step_per_epoch = total_samples // effective_batch_size

    total_step = step_per_epoch * args.num_train_epochs * args.num_generations // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_ratio * total_step,
        num_training_steps=total_step,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    #vae.enable_tiling()

    if rank <= 0:
        project = _reward_project_name(args, "wan2_2")
        wandb.init(project=project, config=args, name=args.exp_name)

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {step_per_epoch}")
    train_param_count = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if transformer_2 is not None:
        train_param_count += sum(p.numel() for p in transformer_2.parameters() if p.requires_grad)
    main_print(
        f"  Total training parameters per FSDP shard = {train_param_count / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    if (
        args.eval_every_steps > 0
        and args.output_dir is not None
        and eval_prompts
    ):
        if ema is not None:
            ema.copy_to(params_to_optimize, store_temp=True)
        run_eval_videos(
            args,
            transformer,
            transformer_2,
            vae,
            eval_prompts,
            device,
            args.output_dir,
            0,
            rank,
            world_size,
        )
        if ema is not None:
            ema.restore(params_to_optimize)
                
    dist.barrier()

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, step_per_epoch * args.num_train_epochs),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )


    step_times = deque(maxlen=100)
    kl_beta = float(getattr(args, "kl_beta", 0.0))
    kl_controller = None
    if getattr(args, "kl_adaptive", False) and rank <= 0:
        kl_controller = AdaptiveKLController(
            init_beta=kl_beta,
            target=float(getattr(args, "kl_target", 0.0)),
            horizon=int(getattr(args, "kl_horizon", 100)),
            beta_min=float(getattr(args, "kl_beta_min", 0.0)),
            beta_max=float(getattr(args, "kl_beta_max", 1e6)),
            ema_alpha=float(getattr(args, "kl_ema_alpha", 0.2)),
        )

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(args.num_train_epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch       
        if epoch > 0:
            epoch_step = epoch * step_per_epoch
            ema_applied = False
            if (
                ema is not None
                and getattr(args, "ema_use_in_checkpoint", True)
                and epoch_step >= args.ema_start_step
            ):
                ema.copy_to(params_to_optimize, store_temp=True)
                ema_applied = True
            if getattr(args, "use_lora", False):
                save_wan_lora_checkpoint(
                    transformer,
                    transformer_2,
                    optimizer,
                    pipe,
                    args.output_dir,
                    epoch_step,
                    epoch - 1,
                    rank,
                    target_components,
                )
            else:
                save_wan_full_checkpoint(
                    transformer,
                    transformer_2,
                    args.output_dir,
                    epoch_step,
                    epoch - 1,
                    rank,
                )
            if ema_applied:
                ema.restore(params_to_optimize)
            dist.barrier()
        for step, (prompt_embeds, negative_prompt_embeds, caption) in enumerate(train_dataloader):
            prompt_embeds = prompt_embeds.to(device)
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            start_time = time.time()
            global_step = epoch * step_per_epoch + step
            if (step-1) % args.checkpointing_steps == 0 and step!=1:
                ema_applied = False
                if (
                    ema is not None
                    and getattr(args, "ema_use_in_checkpoint", True)
                    and global_step >= args.ema_start_step
                ):
                    ema.copy_to(params_to_optimize, store_temp=True)
                    ema_applied = True
                if getattr(args, "use_lora", False):
                    save_wan_lora_checkpoint(
                        transformer,
                        transformer_2,
                        optimizer,
                        pipe,
                        args.output_dir,
                        step,
                        epoch,
                        rank,
                        target_components,
                    )
                else:
                    save_wan_full_checkpoint(
                        transformer,
                        transformer_2,
                        args.output_dir,
                        step,
                        epoch,
                        rank,
                    )
                if ema_applied:
                    ema.restore(params_to_optimize)
                dist.barrier()

            loss, grad_norm, dim_reward, mean_kl_loss = train_one_step(
                args,
                device, 
                transformer,
                transformer_2,
                ref_transformer,
                ref_transformer_2,
                vae,
                reward_dispatcher,
                optimizer,
                lr_scheduler,
                prompt_embeds, 
                negative_prompt_embeds, 
                caption,
                noise_scheduler,
                args.max_grad_norm,
                kl_beta=kl_beta,
                ema=ema,
                ema_parameters=params_to_optimize,
                ema_use_in_rollout=getattr(args, "ema_use_in_rollout", True),
                ema_start_step=args.ema_start_step,
                global_step=global_step,
            )
            if ema is not None and global_step >= args.ema_start_step:
                ema.step(params_to_optimize, global_step)

            if (
        args.eval_every_steps > 0
        and args.output_dir is not None
        and global_step > 0
        and global_step % args.eval_every_steps == 0
    ):
                if eval_prompts:
                    if ema is not None:
                        ema.copy_to(params_to_optimize, store_temp=True)
                    run_eval_videos(
                        args,
                        transformer,
                        transformer_2,
                        vae,
                        eval_prompts,
                        device,
                        args.output_dir,
                        global_step,
                        rank,
                        world_size,
                    )
                    if ema is not None:
                        ema.restore(params_to_optimize)
            dist.barrier()

            if getattr(args, "kl_adaptive", False):
                if rank <= 0 and kl_controller is not None:
                    kl_beta = float(kl_controller.update(mean_kl_loss))
                kl_beta_tensor = torch.tensor([kl_beta], device=device, dtype=torch.float32)
                dist.broadcast(kl_beta_tensor, src=0)
                kl_beta = float(kl_beta_tensor.item())
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                dim_reward_log = {k: np.mean(v) for k, v in dim_reward.items()}
                dim_reward_log.update({f"{k}_std": np.std(v) for k, v in dim_reward.items()})

                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "kl_loss": mean_kl_loss,
                        "kl_beta": kl_beta,
                        **dim_reward_log
                    },
                    step=global_step,
                )



    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


def build_parser():
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Enable EMA tracking for trainable parameters.",
    )
    parser.add_argument(
        "--ema_update_interval",
        type=int,
        default=1,
        help="Number of optimizer steps between EMA updates.",
    )
    parser.add_argument(
        "--ema_use_in_rollout",
        action="store_true",
        default=False,
        help="Use EMA weights during rollout sampling inside training.",
    )
    parser.add_argument(
        "--ema_use_in_checkpoint",
        action="store_true",
        default=False,
        help="Save checkpoints using EMA weights instead of live weights.",
    )
    parser.add_argument(
            "--eval_every_steps",
            type=int,
            default=10,
            help="Run eval every N steps (0 disables).",
        )
    parser.add_argument(
        "--eval_num_prompts",
        type=int,
        default=32,
        help="Number of prompts to sample for eval videos.",
    )
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Enable LoRA fine-tuning (train adapters only).",
    )
    parser.add_argument(
        "--target_components",
        type=str,
        default="transformer,transformer_2",
        help="Comma-separated list of components to train (e.g., transformer,transformer_2).",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="LoRA rank parameter."
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.0,
        help="KL loss coefficient (set > 0 to enable KL regularization against a frozen reference model).",
    )
    parser.add_argument(
        "--kl_adaptive",
        action="store_true",
        default=False,
        help="Enable adaptive KL beta control to keep KL near --kl_target.",
    )
    parser.add_argument(
        "--kl_target",
        type=float,
        default=0.0,
        help="Target mean KL loss used by the adaptive KL controller (requires --kl_adaptive).",
    )
    parser.add_argument(
        "--kl_horizon",
        type=int,
        default=100,
        help="Controller horizon in steps (larger = slower beta updates).",
    )
    parser.add_argument(
        "--kl_beta_min",
        type=float,
        default=0.0,
        help="Minimum KL beta when using --kl_adaptive.",
    )
    parser.add_argument(
        "--kl_beta_max",
        type=float,
        default=1e6,
        help="Maximum KL beta when using --kl_adaptive.",
    )
    parser.add_argument(
        "--kl_ema_alpha",
        type=float,
        default=0.2,
        help="EMA smoothing for KL before updating beta (requires --kl_adaptive).",
    )
    parser.add_argument(
        "--kl_reference_model_name_or_path",
        type=str,
        default=None,
        help="Optional reference model path for KL regularization; if not set, will try to use transformer.disable_adapter() as the reference.",
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help="Resume WAN LoRA training from a previous checkpoint directory.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--apply_gdpo",
        action="store_true",
        default=False,
        help="apply batch normalization to weighted advantages",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--boundary_ratio",
        type=float,
        default=0.875,
        help="Boundary ratio for Wan2.2 dual denoisers; set <0 to disable transformer_2.",
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=1000,
        help="Number of training timesteps used to compute boundary_ratio.",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--cfg_infer",
        type = float,
        default=5.0,
        help="cfg for training",
    )
    parser.add_argument(
        "--cfg_infer_2",
        type=float,
        default=-1.0,
        help="CFG scale for transformer_2 (low-noise stage); set <0 to reuse --cfg_infer.",
    )
    parser.add_argument(
        "--grpo_step_mode",
        type=str,
        default='flow',
        help="flow or dance",
    )
    parser.add_argument(
        "--rationorm",
        action="store_true",
        default=False,
        help="Enable ratio normalization (rationorm) like SD3 training.",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8080",
        help="api address for requesting UnifiedReward",
    )
    parser.add_argument(
        "--reward_spec",
        type=str,
        default=None,
        help="reward spec as JSON or name:weight list",
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.05,
        help="Number of steps ratio for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name in wandb project.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Total number of training epochs.",
    )


    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
