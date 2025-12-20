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
import json
import os
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from torch.utils.data.distributed import DistributedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_qwenimage_rl_datasets import LatentDataset
import torch.distributed as dist
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    resume_lora_optimizer,
)
from fastvideo.utils.logging_ import main_print
from diffusers.image_processor import VaeImageProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
from collections import deque
import numpy as np
from typing import List
from fastvideo.utils.fsdp_util_qwenimage import fsdp_wrapper, FSDPConfig
from contextlib import contextmanager
from safetensors.torch import load_file, save_file
from vllm_utils.vllm_request import evaluate_batch
from fastvideo.rewards.unifiedreward import extract_normalized_rewards
from fastvideo.grpo.kl import compute_kl_loss, disable_lora_adapters
from fastvideo.grpo.steps import dance_grpo_step, flow_grpo_step, sd3_time_shift
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict


def _parse_lora_target_modules(arg: str) -> List[str]:
    if not arg:
        return []
    parts = [p.strip() for p in arg.split(",")]
    return [p for p in parts if p]


def _ensure_only_lora_trainable(transformer):
    for name, param in transformer.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def load_qwenimage_lora_weights(transformer, checkpoint_dir: str):
    candidate_files = [
        "adapter_model.safetensors",
        "pytorch_lora_weights.safetensors",
        "lora.safetensors",
    ]
    weight_path = None
    for fname in candidate_files:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            weight_path = path
            break
    if weight_path is None:
        main_print(f"--> No LoRA weights found in {checkpoint_dir}")
        return

    lora_state_dict = load_file(weight_path)

    if any(k.startswith("transformer.") for k in lora_state_dict.keys()):
        lora_state_dict = {
            k.replace("transformer.", "", 1): v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        try:
            from diffusers.utils import convert_unet_state_dict_to_peft

            lora_state_dict = convert_unet_state_dict_to_peft(lora_state_dict)
        except Exception:
            pass

    incompatible_keys = set_peft_model_state_dict(
        transformer, lora_state_dict, adapter_name="default"
    )
    if incompatible_keys is not None:
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            main_print(
                "Loading adapter weights led to unexpected keys not found in the model: "
                f"{unexpected_keys}."
            )
    if hasattr(transformer, "set_adapter"):
        transformer.set_adapter("default")


def save_qwenimage_lora_checkpoint(transformer, optimizer, rank, output_dir, step, epoch):
    with FSDP.state_dict_type(
        transformer,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state_dict = transformer.state_dict()
        lora_optim_state = FSDP.optim_state_dict(transformer, optimizer)

    if rank > 0:
        return

    save_dir = os.path.join(output_dir, f"lora-checkpoint-{step}-{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    optim_path = os.path.join(save_dir, "lora_optimizer.pt")
    torch.save(lora_optim_state, optim_path)

    transformer_lora_layers = get_peft_model_state_dict(
        model=transformer, state_dict=full_state_dict
    )
    save_file(transformer_lora_layers, os.path.join(save_dir, "adapter_model.safetensors"))

    lora_config = {
        "step": step,
        "lora_params": {
            "lora_rank": getattr(transformer.config, "lora_rank", None),
            "lora_alpha": getattr(transformer.config, "lora_alpha", None),
            "target_modules": getattr(transformer.config, "lora_target_modules", None),
        },
    }
    config_path = os.path.join(save_dir, "lora_config.json")
    with open(config_path, "w") as f:
        json.dump(lora_config, f, indent=4)
    main_print(f"--> LoRA checkpoint saved at {save_dir}")

template = (
    "You are presented with a generated image and its associated text caption. Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n\n"
    "1. Evaluate each word in the caption based on how well it is visually represented in the image. Assign a numerical score to each word using the format:\n"
    "   Word-wise Scores: [[\"word1\", score1], [\"word2\", score2], ..., [\"wordN\", scoreN], [\"[No_mistakes]\", scoreM]]\n"
    "   - A higher score indicates that the word is less well represented in the image.\n"
    "   - The special token [No_mistakes] represents whether all elements in the caption were correctly depicted. A high score suggests no mistakes; a low score suggests missing or incorrect elements.\n\n"
    "2. Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the image matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
    "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "---\n\n"
    "Word-wise Scores: [[\"word1\", score1], ..., [\"[No_mistakes]\", scoreM]]\n\n"
    "Alignment Score (1-5): X\n"
    "Coherence Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\nText Caption: [{prompt}]"
            )
class FSDP_EMA:
    def __init__(self, model, decay, rank):
        self.decay = decay
        self.rank = rank
        self.ema_state_dict_rank0 = {}
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            self.ema_state_dict_rank0 = {k: v.clone() for k, v in state_dict.items()}
            main_print("--> Modern EMA handler initialized on rank 0.")

    def update(self, model):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            for key in self.ema_state_dict_rank0:
                if key in model_state_dict:
                    self.ema_state_dict_rank0[key].copy_(
                        self.decay * self.ema_state_dict_rank0[key] + (1 - self.decay) * model_state_dict[key]
                    )

    @contextmanager
    def use_ema_weights(self, model):
        backup_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        backup_state_dict_rank0 = get_model_state_dict(model, options=backup_options)

        load_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_model_state_dict(
            model,
            model_state_dict=self.ema_state_dict_rank0, 
            options=load_options
        )
        
        try:
            yield
        finally:
            restore_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            set_model_state_dict(
                model,
                model_state_dict=backup_state_dict_rank0, 
                options=restore_options
            )

def save_ema_checkpoint(ema_handler, rank, output_dir, step, epoch, config_dict):
    if rank == 0 and ema_handler is not None:
        ema_checkpoint_path = os.path.join(output_dir, f"checkpoint-ema-{step}-{epoch}")
        os.makedirs(ema_checkpoint_path, exist_ok=True)
        weight_path = os.path.join(ema_checkpoint_path ,
                                   "diffusion_pytorch_model.safetensors")
        save_file(ema_handler.ema_state_dict_rank0, weight_path)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(ema_checkpoint_path, "config.json")
        # save dict as json
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        #torch.save(ema_handler.ema_state_dict_rank0, os.path.join(ema_checkpoint_path, "ema_model.pt"))
        main_print(f"--> EMA checkpoint saved at {ema_checkpoint_path}")


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


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

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        prompt_attention_mask,
        img_shapes,
        txt_seq_lens,
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
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                pred= transformer(
                    hidden_states=z,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
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
        prompt_attention_masks, 
        txt_seq_lens,
        img_shapes,
        transformer,
        timesteps,
        i,
        sigma_schedule,
        return_stats: bool = False,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=latents,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_attention_masks,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=None,
            return_dict=False,
        )[0]
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
    transformer,
    vae,
    encoder_hidden_states, 
    prompt_attention_masks, 
    original_length,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
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
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_prev_sample_mean = [] if getattr(args, "rationorm", False) else None
    all_rewards = []  
    all_txt_seq_lens = []
    all_input_data = []
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, 1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx][:,:original_length[batch_idx]]
        batch_prompt_attention_mask = prompt_attention_masks[batch_idx][:,:original_length[batch_idx]]

        batch_caption = [caption[i] for i in batch_idx]
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), 1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        packed_height = 2 * (int(h) // (8 * 2))
        packed_width = 2 * (int(w) // (8 * 2))
        input_latents_new = pack_latents(input_latents, len(batch_idx), 16, packed_height, packed_width)
        txt_seq_lens =  batch_prompt_attention_mask.sum(dim=1).tolist() 
        img_shapes = [[(1, h // 8 // 2, w // 8// 2)]] 
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            if getattr(args, "rationorm", False):
                z, latents, batch_latents, batch_log_probs, batch_prev_sample_mean = run_sample_step(
                    args,
                    input_latents_new.clone(),
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batch_encoder_hidden_states,
                    batch_prompt_attention_mask,
                    img_shapes,
                    txt_seq_lens,
                    grpo_sample,
                )
                all_prev_sample_mean.append(batch_prev_sample_mean)
            else:
                z, latents, batch_latents, batch_log_probs = run_sample_step(
                    args,
                    input_latents_new.clone(),
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batch_encoder_hidden_states,
                    batch_prompt_attention_mask,
                    img_shapes,
                    txt_seq_lens,
                    grpo_sample,
                )
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_txt_seq_lens.append(torch.tensor(txt_seq_lens))
        vae.enable_tiling()
        
        image_processor = VaeImageProcessor(16)
        rank = int(os.environ["RANK"])

        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                latents = latents.to(vae.dtype)
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                image = vae.decode(latents, return_dict=False)[0][:, :, 0]
                decoded_image = image_processor.postprocess(image)
        os.makedirs("images", exist_ok=True)
        save_path = f"./images/qwenimage_{rank}_{index}.png"
        decoded_image[0].save(save_path)

        if args.use_unifiedreward:
            # Process the prompt
            all_input_data.append({
                "images": [save_path],
                "problem": template.format(prompt=batch_caption[0])
            })

        
    if args.use_unifiedreward:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                all_response = evaluate_batch(all_input_data, api_url=args.api_url)
                alignment_reward, style_reward, dim_reward = extract_normalized_rewards([response['model_output'] for response in all_response]) 

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    if getattr(args, "rationorm", False):
        all_prev_sample_mean = torch.cat(all_prev_sample_mean, dim=0)

    alignment_reward = torch.cat(alignment_reward, dim=0)
    style_reward = torch.cat(style_reward, dim=0)

    all_txt_seq_lens = torch.cat(all_txt_seq_lens, dim=0)
    
    if getattr(args, "rationorm", False):
        return (
            alignment_reward,
            style_reward,
            all_latents,
            all_log_probs,
            all_prev_sample_mean,
            sigma_schedule,
            all_txt_seq_lens,
            dim_reward,
        )
    return alignment_reward, style_reward, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens, dim_reward


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
    ref_transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    prompt_embeds, 
    prompt_attention_masks, 
    caption, 
    original_length,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    ema_handler
):
    total_loss = 0.0
    total_kl_loss = 0.0
    kl_loss_steps = 0

    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(prompt_embeds)
        prompt_attention_masks = repeat_tensor(prompt_attention_masks)
        original_length = repeat_tensor(original_length)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    if getattr(args, "rationorm", False):
        (
            alignment_reward,
            style_reward,
            all_latents,
            all_log_probs,
            all_prev_sample_mean,
            sigma_schedule,
            all_txt_seq_lens,
            dim_reward,
        ) = sample_reference_model(
            args,
            device,
            transformer,
            vae,
            encoder_hidden_states,
            prompt_attention_masks,
            original_length,
            reward_model,
            tokenizer,
            caption,
            preprocess_val,
        )
    else:
        alignment_reward, style_reward, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens, dim_reward = sample_reference_model(
            args,
            device,
            transformer,
            vae,
            encoder_hidden_states,
            prompt_attention_masks,
            original_length,
            reward_model,
            tokenizer,
            caption,
            preprocess_val,
        )
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
        "alignment_reward": alignment_reward.to(torch.float32),
        "style_reward": style_reward.to(torch.float32),
        "txt_seq_lens": all_txt_seq_lens,
        "encoder_hidden_states": encoder_hidden_states,
        "prompt_attention_masks": prompt_attention_masks,
        "original_length": original_length,
    }
    if getattr(args, "rationorm", False):
        samples["prev_sample_mean"] = all_prev_sample_mean[:, :-1]
    gathered_align_reward = gather_tensor(samples["alignment_reward"])
    gathered_style_reward = gather_tensor(samples["style_reward"])

    if dist.get_rank()==0:
        print("gathered_align_reward", gathered_align_reward)
        print("gathered_style_reward", gathered_style_reward)

    #计算advantage
    if args.use_group:
        n = len(samples["alignment_reward"]) // (args.num_generations)
        alignment_advantages = torch.zeros_like(samples["alignment_reward"])
        style_advantages = torch.zeros_like(samples["style_reward"])

        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["alignment_reward"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            alignment_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["style_reward"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            style_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

        
        samples["advantages"] = 0.7*style_advantages + 1.4*alignment_advantages
    else:
        alignment_advantages = (samples["alignment_reward"] - gathered_align_reward.mean())/(gathered_align_reward.std()+1e-8)
        style_advantages = (samples["style_reward"] - gathered_style_reward.mean())/(gathered_style_reward.std()+1e-8)

        samples["advantages"] = 0.7*style_advantages + 1.4*alignment_advantages


    
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
            need_stats = getattr(args, "rationorm", False) or args.kl_beta > 0
            kl_loss = None
            if need_stats:
                new_log_probs, prev_sample_mean, noise_scale, dt = grpo_one_step(
                    args,
                    sample["latents"][:,_],
                    sample["next_latents"][:,_],
                    sample["encoder_hidden_states"][:,:sample['original_length']],
                    sample["prompt_attention_masks"][:,:sample['original_length']],
                    sample["txt_seq_lens"],
                    [[(1, args.h // 8 // 2, args.w // 8// 2)]],
                    transformer,
                    sample["timesteps"][:,_],
                    perms[i][_],
                    sigma_schedule,
                    return_stats=True,
                )
            else:
                new_log_probs = grpo_one_step(
                    args,
                    sample["latents"][:,_],
                    sample["next_latents"][:,_],
                    sample["encoder_hidden_states"][:,:sample['original_length']],
                    sample["prompt_attention_masks"][:,:sample['original_length']],
                    sample["txt_seq_lens"],
                    [[(1, args.h // 8 // 2, args.w // 8// 2)]],
                    transformer,
                    sample["timesteps"][:,_],
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

            if args.kl_beta > 0:
                with torch.no_grad():
                    if ref_transformer is not None:
                        _, prev_sample_mean_ref, noise_scale_ref, dt_ref = grpo_one_step(
                            args,
                            sample["latents"][:,_],
                            sample["next_latents"][:,_],
                            sample["encoder_hidden_states"][:,:sample['original_length']],
                            sample["prompt_attention_masks"][:,:sample['original_length']],
                            sample["txt_seq_lens"],
                            [[(1, args.h // 8 // 2, args.w // 8// 2)]],
                            ref_transformer,
                            sample["timesteps"][:,_],
                            perms[i][_],
                            sigma_schedule,
                            return_stats=True,
                        )
                    else:
                        was_training = transformer.training
                        transformer.eval()
                        try:
                            with disable_lora_adapters(transformer):
                                _, prev_sample_mean_ref, noise_scale_ref, dt_ref = grpo_one_step(
                                    args,
                                    sample["latents"][:,_],
                                    sample["next_latents"][:,_],
                                    sample["encoder_hidden_states"][:,:sample['original_length']],
                                    sample["prompt_attention_masks"][:,:sample['original_length']],
                                    sample["txt_seq_lens"],
                                    [[(1, args.h // 8 // 2, args.w // 8// 2)]],
                                    transformer,
                                    sample["timesteps"][:,_],
                                    perms[i][_],
                                    sigma_schedule,
                                    return_stats=True,
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
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            print("alignment_reward", sample["alignment_reward"].item())
            print("style_reward", sample["style_reward"].item())
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            if args.kl_beta > 0 and kl_loss is not None:
                print("kl_loss", float(kl_loss.detach().item()))
            print("final loss", loss.item())
        dist.barrier()
    mean_kl_loss = total_kl_loss / kl_loss_steps if kl_loss_steps else 0.0
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
    preprocess_val = None
    processor = None
    reward_model=None
    

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenImageTransformerBlock
    transformer = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )

    if getattr(args, "use_lora", False):
        transformer.requires_grad_(False)
        target_modules = _parse_lora_target_modules(args.lora_target_modules)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=target_modules,
        )
        if hasattr(transformer, "add_adapter"):
            transformer.add_adapter(transformer_lora_config)
            if hasattr(transformer, "set_adapter"):
                transformer.set_adapter("default")
        else:
            from peft import get_peft_model

            transformer = get_peft_model(transformer, transformer_lora_config)

        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = target_modules
        _ensure_only_lora_trainable(transformer)

    # Setup FSDP configuration
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,  
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=torch.bfloat16,
        use_device_mesh=False, 
    )
    transformer = fsdp_wrapper(transformer, fsdp_config,)

    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    apply_fsdp_checkpointing(
            transformer, (QwenImageTransformerBlock), args.selective_checkpointing
        )

    from diffusers.models.autoencoders import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model (optional, for KL regularization)
    ref_transformer = None
    if getattr(args, "kl_beta", 0.0) > 0:
        if args.kl_reference_model_name_or_path:
            ref_transformer = QwenImageTransformer2DModel.from_pretrained(
                args.kl_reference_model_name_or_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            ).to(device)
            ref_transformer.requires_grad_(False)
            ref_transformer.eval()
        else:
            assert getattr(args, "use_lora", False), (
                "args.kl_beta > 0 requires either a separate ref_transformer "
                "(set --kl_reference_model_name_or_path) or a model that supports adapter disabling "
                "(enable --use_lora)."
            )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

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
        load_qwenimage_lora_weights(transformer, args.resume_from_lora_checkpoint)
        transformer, optimizer, init_steps = resume_lora_optimizer(
            transformer, args.resume_from_lora_checkpoint, optimizer
        )

    

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=None,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    #vae.enable_tiling()
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
    if rank <= 0:
        project = (
            "unifiedreward_qwenimage_lora"
            if getattr(args, "use_lora", False)
            else "unifiedreward_qwenimage"
        )
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
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

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

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(args.num_train_epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        
        if epoch > 0:
            if getattr(args, "use_lora", False):
                save_qwenimage_lora_checkpoint(
                    transformer,
                    optimizer,
                    rank,
                    args.output_dir,
                    epoch * step_per_epoch,
                    epoch - 1,
                )
            else:
                save_checkpoint(
                    transformer, rank, args.output_dir, epoch * step_per_epoch, epoch - 1
                )
            dist.barrier()

        for step, (prompt_embeds, prompt_attention_masks, caption, original_length) in enumerate(train_dataloader):
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_masks = prompt_attention_masks.to(device)
            start_time = time.time()
            if (step-1) % args.checkpointing_steps == 0 and step!=1:
                if getattr(args, "use_lora", False):
                    save_qwenimage_lora_checkpoint(
                        transformer, optimizer, rank, args.output_dir, step, epoch
                    )
                else:
                    save_checkpoint(transformer, rank, args.output_dir, step, epoch)
                if args.use_ema:
                    save_ema_checkpoint(ema_handler, rank, args.output_dir, step, epoch, dict(transformer.config))


                dist.barrier()

            loss, grad_norm, dim_reward, mean_kl_loss = train_one_step(
                args,
                device, 
                transformer,
                ref_transformer,
                vae,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                prompt_embeds, 
                prompt_attention_masks, 
                caption, 
                original_length,
                noise_scheduler,
                args.max_grad_norm,
                preprocess_val,
                ema_handler,
            )

            if args.use_ema and ema_handler:
                ema_handler.update(transformer)
    
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
                global_step = epoch * step_per_epoch + step
                dim_reward_log = {k: np.mean(v) for k, v in dim_reward.items()}
                dim_reward_log.update({f"{k}_std": np.std(v) for k, v in dim_reward.items()})

                wandb.log(
                    {
                        "train_loss": loss,
                        "kl_loss": mean_kl_loss,
                        "kl_beta": getattr(args, "kl_beta", 0.0),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
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
        "--use_lora",
        action="store_true",
        default=False,
        help="Enable LoRA fine-tuning (train adapters only).",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank (r).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="add_k_proj,add_q_proj,add_v_proj,to_add_out,to_k,to_out.0,to_q,to_v",
        help="Comma-separated target module names for LoRA injection.",
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help="Path to a LoRA checkpoint directory (e.g. lora-checkpoint-STEP-EPOCH).",
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
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_pickscore",
        action="store_true",
        default=False,
        help="whether use pickscore as reward model",
    )
    parser.add_argument(
        "--use_hpsv3",
        action="store_true",
        default=False,
        help="whether use hpsv3 as reward model",
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
        "--kl_beta",
        type=float,
        default=0.0,
        help="KL regularization weight (0 disables KL).",
    )
    parser.add_argument(
        "--kl_reference_model_name_or_path",
        type=str,
        default=None,
        help="Optional reference model path for KL; if omitted, requires --use_lora so adapters can be disabled.",
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Enable Exponential Moving Average of model weights."
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
        "--use_unifiedreward",
        action="store_true",
        default=False,
        help="whether use UnifiedReward as reward model",
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
