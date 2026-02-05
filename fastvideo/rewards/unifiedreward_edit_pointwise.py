from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

import torch

from fastvideo.rewards.templates import (
    get_unifiedreward_edit_pointwise_image_quality_template,
    get_unifiedreward_edit_pointwise_instruction_following_template,
)
from vllm_utils.vllm_request import evaluate_batch


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_quality_weights() -> Tuple[float, float]:
    w0 = _get_env_float("EDIT_QUALITY_WEIGHT_NATURALNESS", 1.0)
    w1 = _get_env_float("EDIT_QUALITY_WEIGHT_ARTIFACTS", 1.0)
    return w0, w1


def get_instruction_weights() -> Tuple[float, float]:
    w0 = _get_env_float("EDIT_IF_WEIGHT_SUCCESS", 1.0)
    w1 = _get_env_float("EDIT_IF_WEIGHT_OVEREDIT", 1.0)
    return w0, w1


def _extract_json(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            return json.dumps(decoded)
        if isinstance(decoded, str):
            text = decoded.strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _parse_scores(text: str) -> List[float]:
    payload = _extract_json(text)
    if payload is None:
        return [0.0, 0.0]
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return [0.0, 0.0]
    if not isinstance(obj, dict):
        return [0.0, 0.0]
    scores = obj.get("score")
    if not isinstance(scores, (list, tuple)) or len(scores) < 2:
        return [0.0, 0.0]
    try:
        return [float(scores[0]), float(scores[1])]
    except (TypeError, ValueError):
        return [0.0, 0.0]


def score_edit_image_quality(
    items: List[dict],
    *,
    api_url: str,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
    template = get_unifiedreward_edit_pointwise_image_quality_template()
    payload = [
        {
            "images": [item["path"]],
            "problem": template,
        }
        for item in items
    ]
    responses = evaluate_batch(payload, api_url=api_url)
    scores = [_parse_scores(resp.get("model_output", "")) for resp in responses]
    dim0_rates = [score[0] for score in scores]
    dim1_rates = [score[1] for score in scores]
    w0, w1 = get_quality_weights()
    denom = w0 + w1
    if denom <= 0:
        denom = 1.0
    combined = [(w0 * a + w1 * b) / denom for a, b in zip(dim0_rates, dim1_rates)]
    dim0_tensor = torch.tensor(dim0_rates, device=device, dtype=torch.float32)
    dim1_tensor = torch.tensor(dim1_rates, device=device, dtype=torch.float32)
    combined_tensor = torch.tensor(combined, device=device, dtype=torch.float32)
    dim_reward = {
        "edit_quality_naturalness": dim0_rates,
        "edit_quality_artifacts": dim1_rates,
    }
    return dim0_tensor, dim1_tensor, combined_tensor, dim_reward


def score_edit_instruction_following(
    items: List[dict],
    *,
    api_url: str,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
    payload = []
    for item in items:
        problem = get_unifiedreward_edit_pointwise_instruction_following_template(
            item.get("instruction", "")
        )
        payload.append(
            {
                "images": [item["source_path"], item["edited_path"]],
                "problem": problem,
            }
        )
    responses = evaluate_batch(payload, api_url=api_url)
    scores = [_parse_scores(resp.get("model_output", "")) for resp in responses]
    dim0_rates = [score[0] for score in scores]
    dim1_rates = [score[1] for score in scores]
    w0, w1 = get_instruction_weights()
    denom = w0 + w1
    if denom <= 0:
        denom = 1.0
    combined = [(w0 * a + w1 * b) / denom for a, b in zip(dim0_rates, dim1_rates)]
    dim0_tensor = torch.tensor(dim0_rates, device=device, dtype=torch.float32)
    dim1_tensor = torch.tensor(dim1_rates, device=device, dtype=torch.float32)
    combined_tensor = torch.tensor(combined, device=device, dtype=torch.float32)
    dim_reward = {
        "edit_if_success": dim0_rates,
        "edit_if_overedit": dim1_rates,
    }
    return dim0_tensor, dim1_tensor, combined_tensor, dim_reward
