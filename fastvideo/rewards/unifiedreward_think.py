import base64
import itertools
import re
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from vllm_utils.vllm_request import evaluate_batch


def _extract_answer(text: str) -> Optional[str]:
    final_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return final_match.group(1).strip() if final_match else None


def _pairwise_win_rate(
    *,
    num_items: int,
    responses: List[dict],
    better_1: str,
    better_2: str,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    win_count = {"overall": defaultdict(int)}
    compare_count = {"overall": defaultdict(int)}

    for result in responses:
        idx1 = result["first_index"]
        idx2 = result["second_index"]

        compare_count["overall"][idx1] += 1
        compare_count["overall"][idx2] += 1

        output = result.get("model_output", "")
        final_conclusion = _extract_answer(output)

        if final_conclusion:
            if better_1 in final_conclusion:
                win_count["overall"][idx1] += 1
            elif better_2 in final_conclusion:
                win_count["overall"][idx2] += 1
            else:
                win_count["overall"][idx1] += 0.5
                win_count["overall"][idx2] += 0.5
        else:
            win_count["overall"][idx1] += 0.5
            win_count["overall"][idx2] += 0.5

    overall_win_rate = [
        torch.tensor(
            round(win_count["overall"][idx] / compare_count["overall"][idx], 3),
            device=device,
        ).unsqueeze(0)
        if compare_count["overall"][idx] > 0
        else torch.tensor(0.0, device=device).unsqueeze(0)
        for idx in range(num_items)
    ]

    dim_reward = {
        "overall_reward": [
            round(
                win_count["overall"].get(idx, 0)
                / max(compare_count["overall"].get(idx, 1), 1),
                3,
            )
            for idx in range(num_items)
        ]
    }
    return overall_win_rate, dim_reward


def cal_win_rate_images(
    all_input_data: List[dict],
    *,
    api_url: str,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    images = [data["images"] for data in all_input_data]
    pairs = list(itertools.combinations(enumerate(images), 2))
    problem = all_input_data[0]["problem"]
    payload = [
        {
            "images": [img1, img2],
            "problem": problem,
            "first_index": idx1,
            "second_index": idx2,
        }
        for (idx1, img1), (idx2, img2) in pairs
    ]

    responses = evaluate_batch(payload, api_url=api_url)
    return _pairwise_win_rate(
        num_items=len(images),
        responses=responses,
        better_1="Image 1 is better",
        better_2="Image 2 is better",
        device=device,
    )


def _encode_image(image: Image.Image) -> str:
    image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _extract_frames(video_path: str, *, num_frames: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return []

    num_frames = min(num_frames, frame_count)
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def cal_win_rate_videos(
    all_input_data: List[dict],
    *,
    api_url: str,
    num_frames: int = 8,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    videos = [
        _extract_frames(data["videos"], num_frames=num_frames) for data in all_input_data
    ]

    pairs = list(itertools.combinations(enumerate(videos), 2))
    problem = all_input_data[0]["problem"]
    payload = [
        {
            "images": [_encode_image(frame) for frame in frames1 + frames2],
            "problem": problem,
            "first_index": idx1,
            "second_index": idx2,
        }
        for (idx1, frames1), (idx2, frames2) in pairs
    ]

    responses = evaluate_batch(payload, api_url=api_url)
    return _pairwise_win_rate(
        num_items=len(videos),
        responses=responses,
        better_1="Video 1 is better",
        better_2="Video 2 is better",
        device=device,
    )

