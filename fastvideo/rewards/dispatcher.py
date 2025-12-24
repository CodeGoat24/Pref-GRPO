from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image

from fastvideo.rewards.clip_reward import compute_clip_score, init_clip_model
from fastvideo.rewards.templates import (
    get_unifiedreward_image_template,
    get_unifiedreward_think_image_template,
    get_unifiedreward_think_video_template,
)
from fastvideo.rewards.unifiedreward import extract_normalized_rewards
from fastvideo.rewards.unifiedreward_think import cal_win_rate_images, cal_win_rate_videos
from vllm_utils.vllm_request import evaluate_batch

SUPPORTED_REWARDS = {
    "clip",
    "unifiedreward_think",
    "unifiedreward_alignment",
    "unifiedreward_style",
}

def parse_reward_spec(spec: str) -> "OrderedDict[str, float]":
    reward_weights: "OrderedDict[str, float]" = OrderedDict()
    if spec is None:
        return reward_weights
    spec = spec.strip()
    if not spec:
        return reward_weights
    if spec.startswith("{"):
        try:
            payload = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON reward_spec: {exc}") from exc
        if isinstance(payload, dict):
            for name, weight in payload.items():
                reward_weights[str(name)] = float(weight)
            return reward_weights
        if isinstance(payload, list):
            for name in payload:
                reward_weights[str(name)] = 1.0
            return reward_weights
        raise ValueError("JSON reward_spec must be an object or list.")
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            name, weight_str = entry.split(":", 1)
            weight = float(weight_str.strip())
        else:
            name = entry
            weight = 1.0
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid reward spec entry: {entry!r}")
        reward_weights[name] = weight
    return reward_weights


def reward_spec_has_any(reward_weights: Dict[str, float], names: Iterable[str]) -> bool:
    return any(name in reward_weights for name in names)


def build_reward_inputs(reward_weights: Dict[str, float]) -> Dict[str, List[dict]]:
    reward_inputs: Dict[str, List[dict]] = {}
    if "clip" in reward_weights:
        reward_inputs["clip"] = []
    if "unifiedreward_think" in reward_weights:
        reward_inputs["unifiedreward_think"] = []
    if reward_spec_has_any(reward_weights, ("unifiedreward_alignment", "unifiedreward_style")):
        reward_inputs["unifiedreward"] = []
    return reward_inputs


class RewardDispatcher:
    def __init__(
        self,
        *,
        args,
        device,
        reward_weights: Dict[str, float],
        modality: str = "image",
        clip_model_name: str = "ViT-H-14",
        clip_pretrained_path: str = "./open_clip_pytorch_model.bin",
        clip_image_loader=None,
    ) -> None:
        if not reward_weights:
            raise ValueError("reward_weights is empty; specify at least one reward.")
        for name in reward_weights:
            if name not in SUPPORTED_REWARDS:
                raise ValueError(f"Unsupported reward name: {name}")
        self.args = args
        self.device = device
        self.reward_weights = reward_weights
        self.modality = modality
        self.clip_image_loader = clip_image_loader
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        if "clip" in reward_weights:
            self.clip_model, self.clip_preprocess, self.clip_tokenizer = init_clip_model(
                device,
                model_name=clip_model_name,
                pretrained_path=clip_pretrained_path,
            )

    def build_reward_inputs(self) -> Dict[str, List[dict]]:
        return build_reward_inputs(self.reward_weights)

    def _load_clip_image(self, path: str) -> Image.Image:
        if self.clip_image_loader is not None:
            return self.clip_image_loader(path)
        return Image.open(path).convert("RGB")

    def compute_rewards(
        self,
        reward_inputs: Dict[str, List[dict]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[float]]]:
        reward_tensors: Dict[str, torch.Tensor] = {}
        dim_reward: Dict[str, List[float]] = {}

        if not reward_inputs or not any(reward_inputs.values()):
            raise ValueError("No reward inputs provided.")

        if "clip" in reward_inputs:
            if (
                self.clip_model is None
                or self.clip_preprocess is None
                or self.clip_tokenizer is None
            ):
                raise ValueError("CLIP reward requested but clip model is not initialized.")
            clip_scores = []
            for item in reward_inputs["clip"]:
                clip_image = self._load_clip_image(item["path"])
                clip_scores.append(
                    compute_clip_score(
                        self.clip_model,
                        self.clip_preprocess,
                        self.clip_tokenizer,
                        clip_image,
                        item["prompt"],
                        self.device,
                    )
                )
            if not clip_scores:
                raise ValueError("No CLIP reward inputs provided.")
            clip_scores_tensor = torch.cat(clip_scores, dim=0)
            reward_tensors["clip"] = clip_scores_tensor
            dim_reward.update({"CLIP_score": clip_scores_tensor.cpu().numpy()})

        if "unifiedreward_think" in reward_inputs:
            if self.modality == "video":
                template = get_unifiedreward_think_video_template()
                all_input_data = [
                    {"videos": item["path"], "problem": template.format(prompt=item["prompt"])}
                    for item in reward_inputs["unifiedreward_think"]
                ]
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        rewards_list, dim_reward_local = cal_win_rate_videos(
                            all_input_data,
                            api_url=self.args.api_url,
                            device=self.device,
                        )
            else:
                template = get_unifiedreward_think_image_template()
                all_input_data = [
                    {"images": item["path"], "problem": template.format(prompt=item["prompt"])}
                    for item in reward_inputs["unifiedreward_think"]
                ]
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        rewards_list, dim_reward_local = cal_win_rate_images(
                            all_input_data,
                            api_url=self.args.api_url,
                            device=self.device,
                        )
            if not rewards_list:
                raise ValueError("No UnifiedReward-Think inputs provided.")
            reward_tensors["unifiedreward_think"] = torch.cat(rewards_list, dim=0)
            dim_reward.update(dim_reward_local)

        if "unifiedreward" in reward_inputs:
            template = get_unifiedreward_image_template()
            all_input_data = [
                {"images": [item["path"]], "problem": template.format(prompt=item["prompt"])}
                for item in reward_inputs["unifiedreward"]
            ]
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    all_response = evaluate_batch(all_input_data, api_url=self.args.api_url)
                    alignment_reward, style_reward, dim_reward_local = extract_normalized_rewards(
                        [response["model_output"] for response in all_response],
                        device=self.device,
                    )
            if not alignment_reward or not style_reward:
                raise ValueError("No UnifiedReward inputs provided.")
            alignment_tensor = torch.cat(alignment_reward, dim=0)
            style_tensor = torch.cat(style_reward, dim=0)
            if "unifiedreward_alignment" in self.reward_weights:
                reward_tensors["unifiedreward_alignment"] = alignment_tensor
            if "unifiedreward_style" in self.reward_weights:
                reward_tensors["unifiedreward_style"] = style_tensor
            dim_reward.update(dim_reward_local)

        return reward_tensors, dim_reward


def compute_weighted_advantages(
    reward_tensors: Dict[str, torch.Tensor],
    reward_weights: Dict[str, float],
    *,
    gather_tensor,
    use_group: bool,
    num_generations: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not reward_tensors:
        raise ValueError("reward_tensors is empty; cannot compute advantages.")
    advantages = torch.zeros_like(next(iter(reward_tensors.values())))
    reward_advantages: Dict[str, torch.Tensor] = {}
    for name in reward_weights.keys():
        if name not in reward_tensors:
            raise ValueError(f"Missing reward tensor for: {name}")
        rewards = reward_tensors[name]
        if use_group:
            n = len(rewards) // num_generations
            adv = torch.zeros_like(rewards)
            for i in range(n):
                start_idx = i * num_generations
                end_idx = (i + 1) * num_generations
                group_rewards = rewards[start_idx:end_idx]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                adv[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        else:
            gathered_reward = gather_tensor(rewards)
            adv = (rewards - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
        reward_advantages[name] = adv
        advantages = advantages + reward_weights.get(name, 0.0) * adv
    return advantages, reward_advantages
