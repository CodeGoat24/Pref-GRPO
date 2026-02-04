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
import os
import random
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
import time
from torch.utils.data import DataLoader
import torch
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.flux2_klein_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from fastvideo.utils.checkpoint import (
    save_checkpoint_ddp,
    save_lora_checkpoint_ddp,
    resume_lora_optimizer_ddp,
)
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.config_io import dump_args_yaml
from fastvideo.utils.rollout_io import save_rollout_image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
from collections import deque
import numpy as np
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from fastvideo.rewards.dispatcher import (
    compute_weighted_advantages,
    parse_reward_spec,
    RewardDispatcher,
)
from fastvideo.grpo.kl import compute_kl_loss, disable_lora_adapters
from fastvideo.grpo.steps import dance_grpo_step, flow_grpo_step, sd3_time_shift
from fastvideo.grpo.ema import EMAModuleWrapper

def _parse_lora_target_modules(arg, default):
    if arg is None:
        return list(default)
    items = [s.strip() for s in str(arg).split(",")]
    items = [s for s in items if s]
    return items if items else list(default)

def _reward_project_name(args, base):
    reward_names = "_".join(sorted(args.reward_weights.keys()))
    suffix = "_lora" if getattr(args, "use_lora", False) else ""
    return f"{base}_{reward_names}{suffix}"

def _clip_grad_norm(model, max_grad_norm):
    if hasattr(model, "clip_grad_norm_"):
        return model.clip_grad_norm_(max_grad_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

def _unwrap_transformer(transformer):
    return transformer.module if isinstance(transformer, DDP) else transformer

def _resolve_rollout_store_device(args, device):
    store_device = device
    if getattr(args, "rollout_store_device", "cuda") == "cpu":
        store_device = torch.device("cpu")
    return store_device

def _load_negative_prompt_embeddings(args, device):
    data_json_path = getattr(args, "data_json_path", None)
    if not data_json_path:
        return None, None
    dataset_dir = os.path.dirname(data_json_path)
    embed_path = os.path.join(dataset_dir, "negative_prompt_embed.pt")
    text_ids_path = os.path.join(dataset_dir, "negative_text_ids.pt")
    if not (os.path.exists(embed_path) and os.path.exists(text_ids_path)):
        return None, None
    negative_prompt_embeds = torch.load(
        embed_path, map_location="cpu", weights_only=True
    )
    negative_text_ids = torch.load(
        text_ids_path, map_location="cpu", weights_only=True
    )
    return negative_prompt_embeds.to(device), negative_text_ids.to(device)

def ensure_train_grads(transformer):
    torch.set_grad_enabled(True)
    if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
        torch._C._set_inference_mode(False)
    if not any(p.requires_grad for p in transformer.parameters()):
        transformer.requires_grad_(True)


def embedding_dataloader_wrapper(dataloader, device):
    while True:
        for prompt_embeds, text_ids, captions in dataloader:
            prompt_embeds = prompt_embeds.to(device)
            text_ids = text_ids.to(device)
            yield prompt_embeds, text_ids, captions


def _extract_caption(data_item):
    for key in ("caption", "prompt", "text"):
        if key in data_item and data_item[key] is not None:
            return str(data_item[key])
    return ""


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
        text_ids_file = data_item["text_ids"]
        prompt_embed = torch.load(
            os.path.join(dataset.prompt_embed_dir, prompt_embed_file),
            map_location="cpu",
            weights_only=True,
        )
        text_ids = torch.load(
            os.path.join(dataset.text_ids_dir, text_ids_file),
            map_location="cpu",
            weights_only=True,
        )
        caption = _extract_caption(data_item)
        samples.append((prompt_embed, text_ids, caption))
    return samples


def prepare_latents(pipeline, batch_size, height, width, device, dtype, generator):
    transformer = _unwrap_transformer(pipeline.transformer)
    num_latents_channels = transformer.config.in_channels // 4
    latents, latent_ids = pipeline.prepare_latents(
        batch_size=batch_size,
        num_latents_channels=num_latents_channels,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    return latents, latent_ids


def decode_latents(pipeline, latents, latent_ids):
    unpacked = pipeline._unpack_latents_with_ids(latents, latent_ids)
    latents_bn_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(
        unpacked.device, unpacked.dtype
    )
    latents_bn_std = torch.sqrt(
        pipeline.vae.bn.running_var.view(1, -1, 1, 1)
        + pipeline.vae.config.batch_norm_eps
    ).to(unpacked.device, unpacked.dtype)
    unpacked = unpacked * latents_bn_std + latents_bn_mean
    unpacked = pipeline._unpatchify_latents(unpacked)
    image = pipeline.vae.decode(unpacked, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)


def set_eval_timesteps(scheduler, num_inference_steps, device, image_seq_len):
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    return scheduler.timesteps


def run_eval_images(
    args,
    pipeline,
    transformer,
    eval_prompts,
    device,
    output_dir,
    step,
    rank,
    world_size,
):
    if not eval_prompts:
        return
    eval_guidance_scale = getattr(args, "eval_guidance_scale", 1.0)
    eval_num_inference_steps = getattr(args, "eval_num_inference_steps", None) or args.sampling_steps
    assigned_indices = [i for i in range(len(eval_prompts)) if i % world_size == rank]
    if not assigned_indices:
        return
    eval_root = os.path.join(output_dir, "eval_image", f"{step}_step")
    os.makedirs(eval_root, exist_ok=True)
    height = args.h
    width = args.w
    pipe = pipeline
    pipe.transformer = transformer
    eval_transformer = _unwrap_transformer(transformer)
    negative_prompt_embeds = None
    negative_text_ids = None
    if eval_guidance_scale > 1.0:
        with torch.no_grad():
            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                prompt="",
                device=torch.device("cpu"),
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=tuple(args.text_encoder_out_layers),
            )
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        negative_text_ids = negative_text_ids.to(device)
    was_training = transformer.training
    transformer.eval()
    for idx in assigned_indices:
        prompt_embed, text_ids, caption = eval_prompts[idx]
        prompt_embeds = prompt_embed.to(device).unsqueeze(0)
        text_ids = text_ids.to(device).unsqueeze(0)
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        latents, latent_ids = prepare_latents(
            pipe, 1, height, width, device, prompt_embeds.dtype, generator
        )
        timesteps = set_eval_timesteps(
            pipe.scheduler, eval_num_inference_steps, device, latents.shape[1]
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in timesteps:
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(eval_transformer.dtype)
                latent_image_ids = latent_ids
                with eval_transformer.cache_context("cond"):
                    noise_pred = eval_transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred[:, : latents.size(1) :]
                if eval_guidance_scale > 1.0:
                    with eval_transformer.cache_context("uncond"):
                        neg_noise_pred = eval_transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                    noise_pred = neg_noise_pred + eval_guidance_scale * (
                        noise_pred - neg_noise_pred
                    )
                latents = pipe.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            images = decode_latents(pipe, latents, latent_ids)
        safe_caption = "".join(
            ch if ("0" <= ch <= "9" or "A" <= ch <= "Z" or "a" <= ch <= "z" or ch in "-_")
            else "_"
            for ch in str(caption)
        )
        safe_caption = safe_caption.encode("ascii", errors="ignore").decode("ascii")
        safe_caption = safe_caption.strip("_")[:60] or "prompt"
        save_path = os.path.join(
            eval_root, f"sample_{idx:02d}_rank{rank}_{safe_caption}.png"
        )
        images[0].save(save_path)
    if was_training:
        transformer.train()
    else:
        transformer.eval()
def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        prompt_embeds,
        text_ids,
        latent_ids,
        grpo_sample,
        rollout_store_device,
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance_scale: float = 1.0,
    ):
    if grpo_sample:
        store_device = rollout_store_device or z.device
        all_latents = [z.detach().to(store_device)]
        all_log_probs = []
        all_prev_sample_mean = [] if getattr(args, "rationorm", False) else None
        for i in progress_bar:  # Add progress bar
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full(
                [prompt_embeds.shape[0]], timestep_value, device=z.device, dtype=torch.long
            )
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                cond_pred = transformer(
                    hidden_states=z,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps/1000,
                    guidance=None,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                if (
                    guidance_scale > 1.0
                    and negative_prompt_embeds is not None
                    and negative_text_ids is not None
                ):
                    neg_embeds = negative_prompt_embeds.to(
                        device=prompt_embeds.device, dtype=prompt_embeds.dtype
                    )
                    neg_text = negative_text_ids.to(device=text_ids.device)
                    if neg_embeds.shape[0] != prompt_embeds.shape[0]:
                        neg_embeds = neg_embeds.repeat(prompt_embeds.shape[0], 1, 1)
                    if neg_text.shape[0] != text_ids.shape[0]:
                        neg_text = neg_text.repeat(text_ids.shape[0], 1, 1)
                    uncond_pred = transformer(
                        hidden_states=z,
                        encoder_hidden_states=neg_embeds,
                        timestep=timesteps/1000,
                        guidance=None,
                        txt_ids=neg_text,
                        img_ids=latent_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                else:
                    pred = cond_pred

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
                else:
                    z, pred_original, log_prob = flow_grpo_step(
                        model_output=pred,
                        latents=z.to(torch.float32),
                        eta=args.eta,
                        sigmas=sigma_schedule,
                        index=i,
                        prev_sample=None,
                    )
            z = z.to(torch.bfloat16)
            all_latents.append(z.detach().to(store_device))
            all_log_probs.append(log_prob.detach().to(store_device))
            if getattr(args, "rationorm", False):
                all_prev_sample_mean.append(prev_sample_mean.detach().to(store_device))
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
            prompt_embeds,
            text_ids,
            latent_ids,
            transformer,
            timesteps,
            i,
            sigma_schedule,
            return_stats: bool = False,
            negative_prompt_embeds=None,
            negative_text_ids=None,
            guidance_scale: float = 1.0,
            enable_grad: bool = True,
):
    transformer.train()
    if latent_ids.dim() == 4:
        latent_ids = latent_ids.squeeze(0)
    if enable_grad:
        ensure_train_grads(transformer)
    if enable_grad:
        with torch.autocast("cuda", torch.bfloat16):
            cond_pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps / 1000,
                guidance=None,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            if (
                guidance_scale > 1.0
                and negative_prompt_embeds is not None
                and negative_text_ids is not None
            ):
                neg_embeds = negative_prompt_embeds.to(
                    device=prompt_embeds.device, dtype=prompt_embeds.dtype
                )
                neg_text = negative_text_ids.to(device=text_ids.device)
                if neg_embeds.shape[0] != prompt_embeds.shape[0]:
                    neg_embeds = neg_embeds.repeat(prompt_embeds.shape[0], 1, 1)
                if neg_text.shape[0] != text_ids.shape[0]:
                    neg_text = neg_text.repeat(text_ids.shape[0], 1, 1)
                uncond_pred = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=neg_embeds,
                    timestep=timesteps / 1000,
                    guidance=None,
                    txt_ids=neg_text,
                    img_ids=latent_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                pred = cond_pred
        ref_param = next((p for p in transformer.parameters() if p.requires_grad), None)
        if ref_param is None:
            raise RuntimeError("No trainable parameters found for gradient attachment.")
        if not pred.requires_grad:
            pred = pred + 0.0 * ref_param.sum()
        assert pred.requires_grad and ref_param.requires_grad, (
            f"pred.requires_grad={pred.requires_grad}, ref_param.requires_grad={ref_param.requires_grad}"
        )
    else:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            cond_pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps / 1000,
                guidance=None,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            if (
                guidance_scale > 1.0
                and negative_prompt_embeds is not None
                and negative_text_ids is not None
            ):
                neg_embeds = negative_prompt_embeds.to(
                    device=prompt_embeds.device, dtype=prompt_embeds.dtype
                )
                neg_text = negative_text_ids.to(device=text_ids.device)
                if neg_embeds.shape[0] != prompt_embeds.shape[0]:
                    neg_embeds = neg_embeds.repeat(prompt_embeds.shape[0], 1, 1)
                if neg_text.shape[0] != text_ids.shape[0]:
                    neg_text = neg_text.repeat(text_ids.shape[0], 1, 1)
                uncond_pred = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=neg_embeds,
                    timestep=timesteps / 1000,
                    guidance=None,
                    txt_ids=neg_text,
                    img_ids=latent_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                pred = cond_pred
    if args.grpo_step_mode == 'dance':    
        z, pred_original, log_prob, prev_sample_mean, noise_scale, dt = dance_grpo_step(
            model_output=pred,
            latents=latents.to(torch.float32),
            eta=args.eta,
            sigmas=sigma_schedule,
            index=i,
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



def sample_reference_model(
    args,
    device, 
    pipeline,
    transformer,
    prompt_embeds,
    text_ids,
    reward_dispatcher,
    caption,
    negative_prompt_embeds=None,
    negative_text_ids=None,
    guidance_scale: float = 1.0,
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

    B = prompt_embeds.shape[0]

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_prev_sample_mean = [] if getattr(args, "rationorm", False) else None
    all_latent_ids = []

    reward_inputs = reward_dispatcher.build_reward_inputs()
    if args.init_same_noise:
        input_latents, latent_ids = prepare_latents(
            pipeline,
            batch_size=1,
            height=h,
            width=w,
            device=device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=device),
        )

    rollout_store_device = _resolve_rollout_store_device(args, device)

    for index, batch_idx in enumerate(batch_indices):
        batch_prompt_embeds = prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        if not args.init_same_noise:
            input_latents, latent_ids = prepare_latents(
                pipeline,
                batch_size=len(batch_idx),
                height=h,
                width=w,
                device=device,
                dtype=torch.bfloat16,
                generator=torch.Generator(device=device),
            )
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            if getattr(args, "rationorm", False):
                z, latents, batch_latents, batch_log_probs, batch_prev_sample_mean = run_sample_step(
                    args,
                    input_latents,
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batch_prompt_embeds,
                    batch_text_ids,
                    latent_ids,
                    grpo_sample,
                    rollout_store_device,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance_scale=guidance_scale,
                )
                all_prev_sample_mean.append(batch_prev_sample_mean)
            else:
                z, latents, batch_latents, batch_log_probs = run_sample_step(
                    args,
                    input_latents,
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batch_prompt_embeds,
                    batch_text_ids,
                    latent_ids,
                    grpo_sample,
                    rollout_store_device,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance_scale=guidance_scale,
                )
        
        all_latent_ids.append(latent_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        rank = int(os.environ["RANK"])

        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded_image = decode_latents(pipeline, latents, latent_ids)
        save_path = save_rollout_image(
            decoded_image[0],
            args.output_dir,
            f"flux2_klein_{rank}_{index}.png",
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

    all_latent_ids = torch.stack(all_latent_ids, dim=0)
    if all_latent_ids.dim() == 4 and all_latent_ids.size(1) == 1:
        all_latent_ids = all_latent_ids.squeeze(1)
    
    
    if getattr(args, "rationorm", False):
        return (
            reward_tensors,
            all_latents,
            all_log_probs,
            all_prev_sample_mean,
            sigma_schedule,
            all_latent_ids,
            dim_reward,
        )
    return reward_tensors, all_latents, all_log_probs, sigma_schedule, all_latent_ids, dim_reward


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    pipeline,
    transformer,
    ref_transformer,
    reward_dispatcher,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    negative_prompt_embeds=None,
    negative_text_ids=None,
):
    # Ensure gradients are enabled for FSDP training forwards.
    ensure_train_grads(transformer)
    total_loss = 0.0
    total_kl_loss = 0.0
    kl_loss_steps = 0
 
    optimizer.zero_grad()
    prompt_embeds, text_ids, caption = next(loader)
    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        prompt_embeds = repeat_tensor(prompt_embeds)
        text_ids = repeat_tensor(text_ids)


        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    if getattr(args, "rationorm", False):
        (
            reward_tensors,
            all_latents,
            all_log_probs,
            all_prev_sample_mean,
            sigma_schedule,
            all_latent_ids,
            dim_reward,
        ) = sample_reference_model(
            args,
            device,
            pipeline,
            transformer,
            prompt_embeds,
            text_ids,
            reward_dispatcher,
            caption,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance_scale=args.train_guidance_scale,
        )
    else:
        reward_tensors, all_latents, all_log_probs, sigma_schedule, all_latent_ids, dim_reward = sample_reference_model(
            args,
            device,
            pipeline,
            transformer,
            prompt_embeds,
            text_ids,
            reward_dispatcher,
            caption,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance_scale=args.train_guidance_scale,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    rollout_device = all_latents.device
    timesteps = torch.tensor(timestep_values, device=rollout_device, dtype=torch.long)

    
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
        "latent_ids": all_latent_ids,
        "text_ids": text_ids,
        "prompt_embeds": prompt_embeds,
    }
    if getattr(args, "rationorm", False):
        samples["prev_sample_mean"] = all_prev_sample_mean[:, :-1]

    
    for name in args.reward_weights.keys():
        rewards = reward_tensors[name]
        gathered_reward = gather_tensor(rewards)
        if dist.get_rank() == 0:
            print(f"gathered_{name}_reward", gathered_reward)
            
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
            torch.randperm(len(samples["timesteps"][0]), device=samples["timesteps"].device)
            for _ in range(batch_size)
        ]
    )
    permute_keys = ["timesteps", "latents", "next_latents", "log_probs"]
    if getattr(args, "rationorm", False):
        permute_keys.append("prev_sample_mean")
    for key in permute_keys:
        samples[key] = samples[key][
            torch.arange(batch_size, device=samples["timesteps"].device)[:, None],
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
            need_stats = getattr(args, "rationorm", False) or args.kl_beta > 0
            latents_step = sample["latents"][:, _]
            next_latents_step = sample["next_latents"][:, _]
            timesteps_step = sample["timesteps"][:, _]
            log_probs_step = sample["log_probs"][:, _]
            if latents_step.device != device:
                latents_step = latents_step.to(device, non_blocking=True)
                next_latents_step = next_latents_step.to(device, non_blocking=True)
                timesteps_step = timesteps_step.to(device, non_blocking=True)
                log_probs_step = log_probs_step.to(device, non_blocking=True)
            step_index = int(perms[i][_].item())
            if need_stats:
                new_log_probs, prev_sample_mean, noise_scale, dt = grpo_one_step(
                    args,
                    latents_step,
                    next_latents_step,
                    sample["prompt_embeds"],
                    sample["text_ids"],
                    sample["latent_ids"],
                    transformer,
                    timesteps_step,
                    step_index,
                    sigma_schedule,
                    return_stats=True,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance_scale=args.train_guidance_scale,
                )
            else:
                new_log_probs = grpo_one_step(
                    args,
                    latents_step,
                    next_latents_step,
                    sample["prompt_embeds"],
                    sample["text_ids"],
                    sample["latent_ids"],
                    transformer,
                    timesteps_step,
                    step_index,
                    sigma_schedule,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance_scale=args.train_guidance_scale,
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

                prev_sample_mean_step = sample["prev_sample_mean"][:, _]
                if prev_sample_mean_step.device != device:
                    prev_sample_mean_step = prev_sample_mean_step.to(device, non_blocking=True)
                diff_sq = (prev_sample_mean.to(torch.float32) - prev_sample_mean_step.to(torch.float32)) ** 2
                reduce_dims = tuple(range(1, diff_sq.ndim))
                ratio_mean_bias = diff_sq.mean(dim=reduce_dims)

                scale = sqrt_dt.mean() * sigma_t
                scale = torch.clamp(scale, min=1e-20)
                ratio_mean_bias = ratio_mean_bias / (2.0 * (scale**2))
                ratio = torch.exp((new_log_probs - log_probs_step + ratio_mean_bias) * scale)
            else:
                ratio = torch.exp(new_log_probs - log_probs_step)

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
            if args.kl_beta > 0:
                with torch.no_grad():
                    if ref_transformer is not None:
                        _, prev_sample_mean_ref, noise_scale_ref, dt_ref = grpo_one_step(
                            args,
                            latents_step,
                            next_latents_step,
                            sample["prompt_embeds"],
                            sample["text_ids"],
                            sample["latent_ids"],
                            ref_transformer,
                            timesteps_step,
                            step_index,
                            sigma_schedule,
                            return_stats=True,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_text_ids=negative_text_ids,
                            guidance_scale=args.train_guidance_scale,
                            enable_grad=False,
                        )
                    else:
                        was_training = transformer.training
                        transformer.eval()
                        try:
                            with disable_lora_adapters(transformer):
                                _, prev_sample_mean_ref, noise_scale_ref, dt_ref = grpo_one_step(
                                    args,
                                    latents_step,
                                    next_latents_step,
                                    sample["prompt_embeds"],
                                    sample["text_ids"],
                                    sample["latent_ids"],
                                    transformer,
                                    timesteps_step,
                                    step_index,
                                    sigma_schedule,
                                    return_stats=True,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    negative_text_ids=negative_text_ids,
                                    guidance_scale=args.train_guidance_scale,
                                    enable_grad=False,
                                )
                        finally:
                            transformer.train(was_training)

                kl_loss = compute_kl_loss(
                    prev_sample_mean, prev_sample_mean_ref, noise_scale_ref
                )
                loss = loss + args.kl_beta * kl_loss

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
            grad_norm = _clip_grad_norm(transformer, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            for name in reward_tensors.keys():
                print(f"{name}_reward", sample[f"reward_{name}"].item())
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            if args.kl_beta > 0 and kl_loss is not None:
                print("kl_loss", float(kl_loss.detach().item()))
            print("final loss", loss.item())
        dist.barrier()
    mean_kl_loss = total_kl_loss / max(kl_loss_steps, 1)
    return total_loss, grad_norm.item(), dim_reward, mean_kl_loss


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(True)
    if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
        torch._C._set_inference_mode(False)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180000))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialize_sequence_parallel_state(args.sp_size)

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
        modality="image",
    )

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )
    transformer = pipeline.transformer

    pipe = None
    target_modules = None
    if getattr(args, "use_lora", False):
        from diffusers.utils import convert_unet_state_dict_to_peft
        from peft import LoraConfig, set_peft_model_state_dict

        pipe = Flux2KleinPipeline
        transformer.requires_grad_(False)

        default_target_modules = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.linear_in",
            "ff.linear_out",
            "ff_context.linear_in",
            "ff_context.linear_out",
            "attn.to_qkv_mlp_proj",
        ]
        target_modules = _parse_lora_target_modules(
            getattr(args, "lora_target_modules", None), default_target_modules
        )

        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)

        if args.resume_from_lora_checkpoint:
            lora_state_dict = pipe.lora_state_dict(args.resume_from_lora_checkpoint)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            try:
                transformer_state_dict = convert_unet_state_dict_to_peft(
                    transformer_state_dict
                )
            except Exception:
                pass
            incompatible_keys = set_peft_model_state_dict(
                transformer, transformer_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    main_print(
                        "Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f"{unexpected_keys}. "
                    )

    if not getattr(args, "use_lora", False):
        transformer.requires_grad_(True)

    if getattr(args, "use_lora", False):
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = target_modules
        # transformer._no_split_modules = [
        #     no_split_module.__name__ for no_split_module in no_split_modules
        # ]

    if args.gradient_checkpointing:
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        elif hasattr(transformer, "gradient_checkpointing_enable"):
            transformer.gradient_checkpointing_enable()
        else:
            main_print("--> gradient checkpointing requested but transformer lacks support.")

    transformer.to(device)
    transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank)

    pipeline.text_encoder.to(device="cpu")
    pipeline.vae.to(device=device, dtype=torch.bfloat16)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.eval()
    pipeline.vae.eval()

    negative_prompt_embeds = None
    negative_text_ids = None
    if getattr(args, "train_guidance_scale", 1.0) > 1.0:
        negative_prompt_embeds, negative_text_ids = _load_negative_prompt_embeddings(
            args, device
        )
        if negative_prompt_embeds is None or negative_text_ids is None:
            with torch.no_grad():
                negative_prompt_embeds, negative_text_ids = pipeline.encode_prompt(
                    prompt="",
                    device=torch.device("cpu"),
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                    text_encoder_out_layers=tuple(args.text_encoder_out_layers),
            )
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_text_ids = negative_text_ids.to(device)
        else:
            main_print("--> loaded negative prompt embeddings from dataset")

    main_print("--> Initializing DDP")
    # Load the reference model
    ref_transformer = None
    if getattr(args, "kl_beta", 0.0) > 0:
        if args.kl_reference_model_name_or_path:
            ref_transformer = Flux2Transformer2DModel.from_pretrained(
                args.kl_reference_model_name_or_path,
                subfolder="transformer",
                torch_dtype=torch.float32,
            ).to(device)
            ref_transformer.requires_grad_(False)
            ref_transformer.eval()
        else:
            assert getattr(args, "use_lora", False), (
                "args.kl_beta > 0 requires either a separate ref_transformer "
                "(set --kl_reference_model_name_or_path) or a model that supports adapter disabling "
                "(enable --use_lora)."
            )

    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    ema = None
    if getattr(args, "use_ema", False):
        ema = EMAModuleWrapper(
            params_to_optimize,
            decay=args.ema_decay,
            update_step_interval=args.ema_update_interval,
            device=device,
        )

    train_dataset = LatentDataset(args.data_json_path, args.cfg, args.num_sample)
    eval_prompts = None
    if args.eval_every_steps > 0 and args.output_dir is not None:
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
    loader = embedding_dataloader_wrapper(train_dataloader, device)
    

    total_samples = len(train_dataloader)
    effective_batch_size = args.train_sp_batch_size * args.sp_size
    step_per_epoch = total_samples // effective_batch_size

    #vae.enable_tiling()

    if rank <= 0:
        project = _reward_project_name(args, "flux")
        wandb.init(project=project, config=args, name=args.exp_name)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
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
        transformer, optimizer, init_steps = resume_lora_optimizer_ddp(
            transformer, args.resume_from_lora_checkpoint, optimizer
        )

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
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {step_per_epoch}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {total_step // args.num_train_epochs}")
    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    if args.eval_every_steps > 0 and args.output_dir is not None and eval_prompts:
        if ema is not None:
            ema.copy_to(params_to_optimize, store_temp=True)
        run_eval_images(
            args,
            pipeline,
            transformer,
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

    step_times = deque(maxlen=100)

    progress_bar = tqdm(
        range(0, step_per_epoch * args.num_train_epochs),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )
    for epoch in range(args.num_train_epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch

        if epoch > 0:
            ema_applied = False
            if (
                ema is not None
                and getattr(args, "ema_use_in_checkpoint", True)
                and epoch * step_per_epoch >= args.ema_start_step
            ):
                ema.copy_to(params_to_optimize, store_temp=True)
                ema_applied = True
            if getattr(args, "use_lora", False):
                save_lora_checkpoint_ddp(
                    transformer,
                    optimizer,
                    rank,
                    args.output_dir,
                    epoch * step_per_epoch,
                    pipe,
                    epoch - 1,
                )
            else:
                save_checkpoint_ddp(
                    transformer, rank, args.output_dir, epoch * step_per_epoch, epoch - 1
                )
            if ema_applied:
                ema.restore(params_to_optimize)
            dist.barrier()
            
        for step in range(init_steps + epoch * step_per_epoch + 1, (epoch+1) * step_per_epoch+1):
            start_time = time.time()
            ensure_train_grads(transformer)
            if step % args.checkpointing_steps == 0:
                transformer.eval()
                dist.barrier()

                ema_applied = False
                if (
                    ema is not None
                    and getattr(args, "ema_use_in_checkpoint", True)
                    and step >= args.ema_start_step
                ):
                    ema.copy_to(params_to_optimize, store_temp=True)
                    ema_applied = True
                if getattr(args, "use_lora", False):
                    save_lora_checkpoint_ddp(
                        transformer,
                        optimizer,
                        rank,
                        args.output_dir,
                        step,
                        pipe,
                        epoch,
                    )
                else:
                    save_checkpoint_ddp(transformer, rank, args.output_dir, step, epoch)
                if ema_applied:
                    ema.restore(params_to_optimize)
                dist.barrier()
                transformer.train()

            loss, grad_norm, dim_reward, mean_kl_loss = train_one_step(
                args,
                device,
                pipeline,
                transformer,
                ref_transformer,
                reward_dispatcher,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                args.max_grad_norm,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )
            if ema is not None and step >= args.ema_start_step:
                ema.step(params_to_optimize, step)
            if (
                args.eval_every_steps > 0
                and args.output_dir is not None
                and eval_prompts
                and step > 0
                and step % args.eval_every_steps == 0
            ):
                if ema is not None:
                    ema.copy_to(params_to_optimize, store_temp=True)
                run_eval_images(
                    args,
                    pipeline,
                    transformer,
                    eval_prompts,
                    device,
                    args.output_dir,
                    step,
                    rank,
                    world_size,
                )
                if ema is not None:
                    ema.restore(params_to_optimize)
                dist.barrier()
    
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
                print(f'avg_step_time: {avg_step_time}')
                
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
                        "kl_beta": getattr(args, "kl_beta", 0.0),
                         **dim_reward_log
                    },
                    step=step,
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
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Max sequence length for the Qwen3 text encoder.",
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
        help="Hidden layers to extract for Qwen3 prompt embeddings.",
    )

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
        help="Number of prompts to sample for eval images.",
    )
    parser.add_argument(
        "--eval_guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for eval sampling.",
    )
    parser.add_argument(
        "--eval_num_inference_steps",
        type=int,
        default=None,
        help="Override eval denoising steps (defaults to sampling_steps).",
    )
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--train_guidance_scale",
        type=float,
        default=1.0,
        help="CFG guidance scale for training rollouts/GRPO.",
    )
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
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name in wandb project.",
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
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated module names for LoRA injection (defaults to Flux preset).",
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help="Path to a LoRA checkpoint directory (e.g. lora-checkpoint-STEP-EPOCH).",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.0,
        help="KL loss coefficient (set > 0 to enable KL regularization against a frozen reference model).",
    )
    parser.add_argument(
        "--kl_reference_model_name_or_path",
        type=str,
        default=None,
        help="Optional reference model path for KL regularization; if not set, will try to use transformer.disable_adapter() as the reference.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="data/logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--num_sample",
        type=int,
        default=None,
        help="Total number of training data.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.05,
        help="Number of steps ratio for the warmup in the lr scheduler.",
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
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

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
    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=720,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=720,   
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
        "--reward_spec",
        type=str,
        default=None,
        help="reward spec as JSON or name:weight list",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8080",
        help="api address for requesting UnifiedReward-Think",
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
        "--rollout_store_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to store rollout latents/log_probs (cpu reduces GPU memory).",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
