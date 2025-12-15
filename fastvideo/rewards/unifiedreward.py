import re
from typing import Dict, List, Tuple

import numpy as np
import torch


def extract_normalized_rewards(
    sample_list: List[str],
    *,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, List[float]]]:
    pattern = r"(\w+) Score \(1-5\):\s*([0-5](?:\.\d+)?)"

    all_scores = []
    for response in sample_list:
        matches = re.findall(pattern, response)
        scores = {key: float(value) for key, value in matches}
        if "Coherence" in scores:
            del scores["Coherence"]
        all_scores.append(scores)

    if not all_scores:
        return [], [], {}

    keys = set()
    for score_dict in all_scores:
        keys.update(score_dict.keys())
    keys = sorted(keys)

    dim_scores_raw = {k: [s[k] for s in all_scores if k in s] for k in keys}
    dim_means = {
        k: np.mean(v) if len(v) > 0 else 0.0 for k, v in dim_scores_raw.items()
    }

    alignment_scores = []
    style_scores = []
    log_alignment_scores = []
    log_style_scores = []

    for score_dict in all_scores:
        alignment_score = score_dict.get("Alignment", dim_means.get("Alignment", 0.0))
        style_score = score_dict.get("Style", dim_means.get("Style", 0.0))

        alignment_scores.append(torch.tensor(alignment_score, device=device).unsqueeze(0))
        style_scores.append(torch.tensor(style_score, device=device).unsqueeze(0))

        log_alignment_scores.append(float(alignment_score))
        log_style_scores.append(float(style_score))

    dim_array = {"Alignment": log_alignment_scores, "Style": log_style_scores}
    return alignment_scores, style_scores, dim_array

