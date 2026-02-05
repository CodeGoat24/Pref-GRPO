from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image

from fastvideo.rewards.aesthetic import AestheticScorer
from fastvideo.rewards.clip_reward import compute_clip_score, init_clip_model
from fastvideo.rewards.reward_paths import REWARD_MODEL_PATHS
from fastvideo.rewards.templates import (
    get_unifiedreward_edit_pairwise_template,
    get_unifiedreward_edit_pointwise_image_quality_template,
    get_unifiedreward_edit_pointwise_instruction_following_template,
    get_unifiedreward_flex_image_template,
    get_unifiedreward_flex_video_template,
    get_unifiedreward_image_template,
    get_unifiedreward_think_image_template,
    get_unifiedreward_think_video_template,
)
from fastvideo.rewards.unifiedreward_edit_pairwise import cal_win_rate_edit_images
from fastvideo.rewards.unifiedreward_edit_pointwise import (
    get_instruction_weights,
    get_quality_weights,
    score_edit_image_quality,
    score_edit_instruction_following,
)
from fastvideo.rewards.unifiedreward import extract_normalized_rewards
from fastvideo.rewards.unifiedreward_flex import (
    _get_env_float as _flex_get_env_float,
    cal_win_rate_images as cal_flex_win_rate_images,
    cal_win_rate_videos as cal_flex_win_rate_videos,
)
from fastvideo.rewards.unifiedreward_think import cal_win_rate_images, cal_win_rate_videos
from vllm_utils.vllm_request import evaluate_batch

SUPPORTED_REWARDS = {
    "aesthetic",
    "clip",
    "hpsv2",
    "hpsv3",
    "pickscore",
    "unifiedreward_edit_pairwise",
    "unifiedreward_edit_pointwise_image_quality",
    "unifiedreward_edit_pointwise_instruction_following",
    "unifiedreward_flex",
    "unifiedreward_think",
    "unifiedreward_alignment",
    "unifiedreward_style",
    "unifiedreward_coherence",
    "videoalign",
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
    if "aesthetic" in reward_weights:
        reward_inputs["aesthetic"] = []
    if "clip" in reward_weights:
        reward_inputs["clip"] = []
    if "hpsv2" in reward_weights:
        reward_inputs["hpsv2"] = []
    if "hpsv3" in reward_weights:
        reward_inputs["hpsv3"] = []
    if "pickscore" in reward_weights:
        reward_inputs["pickscore"] = []
    if "unifiedreward_think" in reward_weights:
        reward_inputs["unifiedreward_think"] = []
    if "unifiedreward_edit_pairwise" in reward_weights:
        reward_inputs["unifiedreward_edit_pairwise"] = []
    if "unifiedreward_edit_pointwise_image_quality" in reward_weights:
        reward_inputs["unifiedreward_edit_pointwise_image_quality"] = []
    if "unifiedreward_edit_pointwise_instruction_following" in reward_weights:
        reward_inputs["unifiedreward_edit_pointwise_instruction_following"] = []
    if "unifiedreward_flex" in reward_weights:
        reward_inputs["unifiedreward_flex"] = []
    if reward_spec_has_any(
        reward_weights,
        ("unifiedreward_alignment", "unifiedreward_style", "unifiedreward_coherence"),
    ):
        reward_inputs["unifiedreward"] = []
    if "videoalign" in reward_weights:
        reward_inputs["videoalign"] = []
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
        clip_pretrained_path: str = REWARD_MODEL_PATHS["clip_pretrained"],
        hps_clip_model_name: str = "ViT-H-14",
        hps_ckpt_path: str = REWARD_MODEL_PATHS["hpsv2_ckpt"],
        pickscore_processor_name: str = REWARD_MODEL_PATHS["pickscore_processor"],
        pickscore_model_name: str = REWARD_MODEL_PATHS["pickscore_model"],
        videoalign_ckpt_path: str = REWARD_MODEL_PATHS["videoalign_ckpt"],
        videoalign_use_norm: bool = True,
        clip_image_loader=None,
    ) -> None:
        if not reward_weights:
            raise ValueError("reward_weights is empty; specify at least one reward.")
        for name in reward_weights:
            if name not in SUPPORTED_REWARDS:
                raise ValueError(f"Unsupported reward name: {name}")
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.args = args
        self.device = device
        self.reward_weights = reward_weights
        self.modality = modality
        self.clip_image_loader = clip_image_loader
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.hps_model = None
        self.hps_preprocess = None
        self.hps_tokenizer = None
        self.hpsv3_inferencer = None
        self.pickscore_model = None
        self.pickscore_processor = None
        self.aesthetic_scorer = None
        self.videoalign_inferencer = None
        self.videoalign_use_norm = videoalign_use_norm
        if "aesthetic" in reward_weights:
            checkpoint_path = REWARD_MODEL_PATHS["aesthetic_ckpt"]
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Aesthetic checkpoint not found: {checkpoint_path}"
                )
            self.aesthetic_scorer = AestheticScorer(
                device=self.device,
                checkpoint_path=checkpoint_path,
                dtype=torch.float32,
            )
        if "clip" in reward_weights:
            self.clip_model, self.clip_preprocess, self.clip_tokenizer = init_clip_model(
                device,
                model_name=clip_model_name,
                pretrained_path=clip_pretrained_path,
            )
        if "hpsv2" in reward_weights:
            try:
                from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
            except ImportError as exc:
                raise ImportError(
                    "hpsv2 is not available; install it before using the hpsv2 reward."
                ) from exc
            model, _, preprocess_val = create_model_and_transforms(
                hps_clip_model_name,
                clip_pretrained_path,
                precision="amp",
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False,
            )
            checkpoint = torch.load(hps_ckpt_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            self.hps_model = model.to(self.device).eval()
            self.hps_preprocess = preprocess_val
            self.hps_tokenizer = get_tokenizer(hps_clip_model_name)
        if "pickscore" in reward_weights:
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "transformers is not available; install it before using pickscore."
                ) from exc
            self.pickscore_processor = AutoProcessor.from_pretrained(pickscore_processor_name)
            self.pickscore_model = (
                AutoModel.from_pretrained(pickscore_model_name).eval().to(device)
            )
        if "hpsv3" in reward_weights:
            try:
                from fastvideo.models.hpsv3 import HPSv3RewardInferencer
            except ImportError as exc:
                raise ImportError(
                    "hpsv3 is not available; install it before using the hpsv3 reward."
                ) from exc
            device_str = str(self.device)
            self.hpsv3_inferencer = HPSv3RewardInferencer(
                checkpoint_path=REWARD_MODEL_PATHS["hpsv3_ckpt"],
                device=device_str,
            )
        if "videoalign" in reward_weights:
            try:
                from fastvideo.models.videoalign.inference import VideoVLMRewardInference
            except ImportError as exc:
                raise ImportError(
                    "videoalign is not available; check fastvideo.models.videoalign."
                ) from exc
            device_str = str(self.device)
            dtype = torch.bfloat16 if "cuda" in device_str else torch.float32
            self.videoalign_inferencer = VideoVLMRewardInference(
                videoalign_ckpt_path,
                device=device_str,
                dtype=dtype,
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

        if "aesthetic" in reward_inputs:
            if self.aesthetic_scorer is None:
                raise ValueError("Aesthetic reward requested but model is not initialized.")
            if self.modality != "image":
                raise ValueError("Aesthetic reward only supports image modality.")
            images = [self._load_clip_image(item["path"]) for item in reward_inputs["aesthetic"]]
            if not images:
                raise ValueError("No aesthetic reward inputs provided.")
            scores_tensor = self.aesthetic_scorer(images).reshape(-1).contiguous()
            reward_tensors["aesthetic"] = scores_tensor
            dim_reward.update({"aesthetic_score": scores_tensor.detach().cpu().numpy()})

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

        if "hpsv2" in reward_inputs:
            if (
                self.hps_model is None
                or self.hps_preprocess is None
                or self.hps_tokenizer is None
            ):
                raise ValueError("HPSv2 reward requested but model is not initialized.")
            if self.modality != "image":
                raise ValueError("HPSv2 reward only supports image modality.")
            hps_scores = []
            for item in reward_inputs["hpsv2"]:
                image = self._load_clip_image(item["path"])
                image_tensor = self.hps_preprocess(image).unsqueeze(0).to(
                    device=self.device, non_blocking=True
                )
                text = self.hps_tokenizer([item["prompt"]]).to(
                    device=self.device, non_blocking=True
                )
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        outputs = self.hps_model(image_tensor, text)
                        image_features = outputs["image_features"]
                        text_features = outputs["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_score = torch.diagonal(logits_per_image)
                hps_scores.append(hps_score)
            if not hps_scores:
                raise ValueError("No HPSv2 reward inputs provided.")
            hps_scores_tensor = torch.cat(hps_scores, dim=0)
            reward_tensors["hpsv2"] = hps_scores_tensor
            dim_reward.update({"hpsv2_score": hps_scores_tensor.cpu().numpy()})

        if "hpsv3" in reward_inputs:
            if self.hpsv3_inferencer is None:
                raise ValueError("HPSv3 reward requested but model is not initialized.")
            if self.modality != "image":
                raise ValueError("HPSv3 reward only supports image modality.")
            image_paths = [
                os.path.abspath(item["path"]) for item in reward_inputs["hpsv3"]
            ]
            prompts = [item["prompt"] for item in reward_inputs["hpsv3"]]
            with torch.no_grad():
                scores = self.hpsv3_inferencer.reward(prompts, image_paths)
            scores_tensor = torch.as_tensor(scores, device=self.device, dtype=torch.float32)
            if scores_tensor.ndim == 2:
                scores_tensor = scores_tensor[:, 0]
            scores_tensor = scores_tensor.reshape(-1).contiguous()
            reward_tensors["hpsv3"] = scores_tensor
            dim_reward.update({"hpsv3_score": scores_tensor.cpu().numpy()})

        if "pickscore" in reward_inputs:
            if self.pickscore_model is None or self.pickscore_processor is None:
                raise ValueError("PickScore reward requested but model is not initialized.")
            if self.modality != "image":
                raise ValueError("PickScore reward only supports image modality.")
            pick_scores = []
            for item in reward_inputs["pickscore"]:
                image = self._load_clip_image(item["path"])
                image_inputs = self.pickscore_processor(
                    images=image,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                text_inputs = self.pickscore_processor(
                    text=item["prompt"],
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    image_embs = self.pickscore_model.get_image_features(**image_inputs)
                    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                    text_embs = self.pickscore_model.get_text_features(**text_inputs)
                    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                    score = (text_embs @ image_embs.T)[0]
                pick_scores.append(score)
            if not pick_scores:
                raise ValueError("No PickScore reward inputs provided.")
            pick_scores_tensor = torch.cat(pick_scores, dim=0)
            reward_tensors["pickscore"] = pick_scores_tensor
            dim_reward.update({"pickscore": pick_scores_tensor.cpu().numpy()})

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

        if "unifiedreward_edit_pairwise" in reward_inputs:
            if self.modality != "image":
                raise ValueError("UnifiedReward edit pairwise only supports image modality.")
            template = get_unifiedreward_edit_pairwise_template()
            all_input_data = []
            for item in reward_inputs["unifiedreward_edit_pairwise"]:
                source_path = item.get("source_path")
                edited_path = item.get("edited_path") or item.get("path")
                instruction = item.get("instruction") or item.get("prompt") or ""
                if not source_path or not edited_path:
                    raise ValueError("Edit pairwise reward requires source_path and edited image path.")
                all_input_data.append(
                    {
                        "source_path": source_path,
                        "edited_path": edited_path,
                        "problem": template.format(instruction=instruction),
                    }
                )
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    rewards_list, dim_reward_local = cal_win_rate_edit_images(
                        all_input_data,
                        api_url=self.args.api_url,
                        device=self.device,
                    )
            reward_tensors["unifiedreward_edit_pairwise"] = torch.cat(rewards_list, dim=0)
            dim_reward.update(dim_reward_local)

        if "unifiedreward_edit_pointwise_image_quality" in reward_inputs:
            if self.modality != "image":
                raise ValueError("UnifiedReward edit pointwise only supports image modality.")
            items = reward_inputs["unifiedreward_edit_pointwise_image_quality"]
            dim0, dim1, combined, dim_local = score_edit_image_quality(
                items,
                api_url=self.args.api_url,
                device=self.device,
            )
            reward_tensors["unifiedreward_edit_pointwise_image_quality_dim0"] = dim0
            reward_tensors["unifiedreward_edit_pointwise_image_quality_dim1"] = dim1
            reward_tensors["unifiedreward_edit_pointwise_image_quality"] = combined
            dim_reward.update(dim_local)

        if "unifiedreward_edit_pointwise_instruction_following" in reward_inputs:
            if self.modality != "image":
                raise ValueError("UnifiedReward edit pointwise only supports image modality.")
            items = reward_inputs["unifiedreward_edit_pointwise_instruction_following"]
            dim0, dim1, combined, dim_local = score_edit_instruction_following(
                items,
                api_url=self.args.api_url,
                device=self.device,
            )
            reward_tensors["unifiedreward_edit_pointwise_instruction_following_dim0"] = dim0
            reward_tensors["unifiedreward_edit_pointwise_instruction_following_dim1"] = dim1
            reward_tensors["unifiedreward_edit_pointwise_instruction_following"] = combined
            dim_reward.update(dim_local)

        if "unifiedreward" in reward_inputs:
            template = get_unifiedreward_image_template()
            all_input_data = [
                {"images": [item["path"]], "problem": template.format(prompt=item["prompt"])}
                for item in reward_inputs["unifiedreward"]
            ]
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    all_response = evaluate_batch(all_input_data, api_url=self.args.api_url)
                    (
                        alignment_reward,
                        style_reward,
                        coherence_reward,
                        dim_reward_local,
                    ) = extract_normalized_rewards(
                        [response["model_output"] for response in all_response],
                        device=self.device,
                    )
            if not alignment_reward or not style_reward or not coherence_reward:
                raise ValueError("No UnifiedReward inputs provided.")
            alignment_tensor = torch.cat(alignment_reward, dim=0)
            style_tensor = torch.cat(style_reward, dim=0)
            coherence_tensor = torch.cat(coherence_reward, dim=0)
            if "unifiedreward_alignment" in self.reward_weights:
                reward_tensors["unifiedreward_alignment"] = alignment_tensor
            if "unifiedreward_style" in self.reward_weights:
                reward_tensors["unifiedreward_style"] = style_tensor
            if "unifiedreward_coherence" in self.reward_weights:
                reward_tensors["unifiedreward_coherence"] = coherence_tensor
            dim_reward.update(dim_reward_local)

        if "unifiedreward_flex" in reward_inputs:
            if self.modality == "video":
                template = get_unifiedreward_flex_video_template()
                all_input_data = [
                    {"videos": item["path"], "problem": template.format(prompt=item["prompt"])}
                    for item in reward_inputs["unifiedreward_flex"]
                ]
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        rewards_list, dim_reward_local = cal_flex_win_rate_videos(
                            all_input_data,
                            api_url=self.args.api_url,
                            device=self.device,
                        )
            elif self.modality == "image":
                template = get_unifiedreward_flex_image_template()
                all_input_data = [
                    {"images": item["path"], "problem": template.format(prompt=item["prompt"])}
                    for item in reward_inputs["unifiedreward_flex"]
                ]
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        rewards_list, dim_reward_local = cal_flex_win_rate_images(
                            all_input_data,
                            api_url=self.args.api_url,
                            device=self.device,
                        )
            else:
                raise ValueError(f"Unsupported modality for UnifiedReward-Flex: {self.modality}")
            if not rewards_list:
                raise ValueError("No UnifiedReward-Flex inputs provided.")
            reward_tensors["unifiedreward_flex"] = torch.cat(rewards_list, dim=0)
            if getattr(self.args, "apply_gdpo", False):
                overall = torch.tensor(
                    dim_reward_local.get("overall_reward", []),
                    device=self.device,
                    dtype=reward_tensors["unifiedreward_flex"].dtype,
                )
                dim_mean = torch.tensor(
                    dim_reward_local.get("dim_mean_reward", []),
                    device=self.device,
                    dtype=reward_tensors["unifiedreward_flex"].dtype,
                )
                dim_flags = dim_reward_local.get(
                    "dim_rate_flags",
                    [True] * len(reward_tensors["unifiedreward_flex"]),
                )
                dim_mask = torch.tensor(dim_flags, device=self.device, dtype=torch.bool)
                reward_tensors["unifiedreward_flex_overall"] = overall
                reward_tensors["unifiedreward_flex_dim_mean"] = dim_mean
                reward_tensors["unifiedreward_flex_dim_mask"] = dim_mask
            dim_reward.update(dim_reward_local)

        if "videoalign" in reward_inputs:
            if self.videoalign_inferencer is None:
                raise ValueError("VideoAlign reward requested but model is not initialized.")
            if self.modality != "video":
                raise ValueError("VideoAlign reward only supports video modality.")
            video_paths = [
                os.path.abspath(item["path"]) for item in reward_inputs["videoalign"]
            ]
            prompts = [item["prompt"] for item in reward_inputs["videoalign"]]
            with torch.no_grad():
                rewards = self.videoalign_inferencer.reward(
                    video_paths,
                    prompts,
                    use_norm=self.videoalign_use_norm,
                )
            if not rewards:
                raise ValueError("No VideoAlign reward inputs provided.")
            vq_scores = [float(r.get("VQ", 0.0)) for r in rewards]
            mq_scores = [float(r.get("MQ", 0.0)) for r in rewards]
            ta_scores = [float(r.get("TA", 0.0)) for r in rewards]
            overall_scores = [
                float(r.get("Overall", vq + mq + ta))
                for r, vq, mq, ta in zip(rewards, vq_scores, mq_scores, ta_scores)
            ]
            reward_tensors["videoalign"] = torch.tensor(
                overall_scores, device=self.device
            )
            dim_reward.update(
                {
                    "videoalign_overall": overall_scores,
                    "videoalign_vq": vq_scores,
                    "videoalign_mq": mq_scores,
                    "videoalign_ta": ta_scores,
                }
            )

        return reward_tensors, dim_reward


def compute_weighted_advantages(
    reward_tensors: Dict[str, torch.Tensor],
    reward_weights: Dict[str, float],
    *,
    gather_tensor,
    use_group: bool,
    num_generations: int,
    apply_gdpo: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not reward_tensors:
        raise ValueError("reward_tensors is empty; cannot compute advantages.")
    advantages = torch.zeros_like(next(iter(reward_tensors.values())))
    reward_advantages: Dict[str, torch.Tensor] = {}

    def _compute_adv(rewards: torch.Tensor) -> torch.Tensor:
        if use_group:
            n = len(rewards) // num_generations
            adv_local = torch.zeros_like(rewards)
            for i in range(n):
                start_idx = i * num_generations
                end_idx = (i + 1) * num_generations
                group_rewards = rewards[start_idx:end_idx]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                adv_local[start_idx:end_idx] = (group_rewards - group_mean) / group_std
            return adv_local
        gathered_reward = gather_tensor(rewards)
        return (rewards - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)

    for name in reward_weights.keys():
        if name not in reward_tensors:
            raise ValueError(f"Missing reward tensor for: {name}")
        rewards = reward_tensors[name]
        if apply_gdpo and name == "unifiedreward_flex":
            overall_key = "unifiedreward_flex_overall"
            dim_key = "unifiedreward_flex_dim_mean"
            mask_key = "unifiedreward_flex_dim_mask"
            if (
                overall_key in reward_tensors
                and dim_key in reward_tensors
                and mask_key in reward_tensors
            ):
                overall_adv = _compute_adv(reward_tensors[overall_key])
                dim_adv = _compute_adv(reward_tensors[dim_key])
                overall_weight = _flex_get_env_float("OVERALL_WEIGHT", 1.0)
                dim_weight = _flex_get_env_float("DIM_WEIGHT", 1.0)
                if dim_weight > 0:
                    blended = (overall_weight * overall_adv + dim_weight * dim_adv) / (
                        overall_weight + dim_weight
                    )
                    mask = reward_tensors[mask_key].to(dtype=torch.bool)
                    adv = torch.where(mask, blended, overall_adv)
                else:
                    adv = overall_adv
            else:
                adv = _compute_adv(rewards)
        elif name == "unifiedreward_edit_pointwise_image_quality":
            dim0_key = "unifiedreward_edit_pointwise_image_quality_dim0"
            dim1_key = "unifiedreward_edit_pointwise_image_quality_dim1"
            if dim0_key not in reward_tensors or dim1_key not in reward_tensors:
                raise ValueError("Missing edit image quality dimension rewards.")
            w0, w1 = get_quality_weights()
            denom = w0 + w1
            if denom <= 0:
                denom = 1.0
            adv0 = _compute_adv(reward_tensors[dim0_key])
            adv1 = _compute_adv(reward_tensors[dim1_key])
            adv = (w0 * adv0 + w1 * adv1) / denom
        elif name == "unifiedreward_edit_pointwise_instruction_following":
            dim0_key = "unifiedreward_edit_pointwise_instruction_following_dim0"
            dim1_key = "unifiedreward_edit_pointwise_instruction_following_dim1"
            if dim0_key not in reward_tensors or dim1_key not in reward_tensors:
                raise ValueError("Missing edit instruction-following dimension rewards.")
            w0, w1 = get_instruction_weights()
            denom = w0 + w1
            if denom <= 0:
                denom = 1.0
            adv0 = _compute_adv(reward_tensors[dim0_key])
            adv1 = _compute_adv(reward_tensors[dim1_key])
            adv = (w0 * adv0 + w1 * adv1) / denom
        else:
            adv = _compute_adv(rewards)
        reward_advantages[name] = adv
        advantages = advantages + reward_weights.get(name, 0.0) * adv
    if apply_gdpo:
        gathered_advantages = gather_tensor(advantages)
        advantages = (advantages - gathered_advantages.mean()) / (
            gathered_advantages.std() + 1e-8
        )
    return advantages, reward_advantages
