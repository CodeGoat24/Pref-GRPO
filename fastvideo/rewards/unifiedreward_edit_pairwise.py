from __future__ import annotations

import itertools
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from vllm_utils.vllm_request import evaluate_batch


def _extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _contains_label(text: str, label: str) -> bool:
    return label.lower() in text.lower()


def _pairwise_win_rate(
    *,
    num_items: int,
    responses: List[dict],
    better_1: str,
    better_2: str,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    win_count = defaultdict(float)
    compare_count = defaultdict(int)

    for result in responses:
        idx1 = result["first_index"]
        idx2 = result["second_index"]

        compare_count[idx1] += 1
        compare_count[idx2] += 1

        output = result.get("model_output", "")
        final_conclusion = _extract_answer(output)

        if final_conclusion:
            if _contains_label(final_conclusion, better_1):
                win_count[idx1] += 1.0
            elif _contains_label(final_conclusion, better_2):
                win_count[idx2] += 1.0
            elif _contains_label(output, better_1):
                win_count[idx1] += 1.0
            elif _contains_label(output, better_2):
                win_count[idx2] += 1.0
            else:
                win_count[idx1] += 0.5
                win_count[idx2] += 0.5
        else:
            win_count[idx1] += 0.5
            win_count[idx2] += 0.5

    overall_win_rate = [
        torch.tensor(
            round(win_count.get(idx, 0.0) / compare_count.get(idx, 1), 3),
            device=device,
        ).unsqueeze(0)
        if compare_count.get(idx, 0) > 0
        else torch.tensor(0.0, device=device).unsqueeze(0)
        for idx in range(num_items)
    ]

    dim_reward = {
        "overall_reward": [
            round(win_count.get(idx, 0.0) / max(compare_count.get(idx, 1), 1), 3)
            for idx in range(num_items)
        ]
    }
    return overall_win_rate, dim_reward


def _group_edit_inputs(all_input_data: List[dict]) -> Dict[tuple, List[tuple]]:
    groups: Dict[tuple, List[tuple]] = defaultdict(list)
    for idx, item in enumerate(all_input_data):
        source_path = item["source_path"]
        problem = item["problem"]
        edited_path = item["edited_path"]
        groups[(source_path, problem)].append((idx, edited_path))
    return groups


def cal_win_rate_edit_images(
    all_input_data: List[dict],
    *,
    api_url: str,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    if not all_input_data:
        raise ValueError("No edit pairwise inputs provided.")

    num_items = len(all_input_data)
    rewards_list = [
        torch.tensor(0.0, device=device).unsqueeze(0) for _ in range(num_items)
    ]
    dim_reward: Dict[str, List[float]] = {"overall_reward": [0.0] * num_items}

    groups = _group_edit_inputs(all_input_data)
    for (source_path, problem), items in groups.items():
        if len(items) < 2:
            continue
        local_entries = list(enumerate(items))
        pairs = list(itertools.combinations(local_entries, 2))
        payload = [
            {
                "images": [source_path, left_item[1], right_item[1]],
                "problem": problem,
                "first_index": left_idx,
                "second_index": right_idx,
            }
            for (left_idx, left_item), (right_idx, right_item) in pairs
        ]

        responses = evaluate_batch(payload, api_url=api_url)
        group_rewards, group_dim = _pairwise_win_rate(
            num_items=len(items),
            responses=responses,
            better_1="Edited image 1 is better",
            better_2="Edited image 2 is better",
            device=device,
        )
        for local_idx, (global_idx, _) in enumerate(items):
            rewards_list[global_idx] = group_rewards[local_idx]
            dim_reward["overall_reward"][global_idx] = group_dim["overall_reward"][
                local_idx
            ]

    return rewards_list, dim_reward
