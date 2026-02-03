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

import json
import os
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def _extract_caption(data_item: dict) -> str:
    for key in ("caption", "prompt", "text"):
        if key in data_item and data_item[key] is not None:
            return str(data_item[key])
    return ""


class LatentDataset(Dataset):
    def __init__(self, json_path: str, cfg_rate: float, num_sample: Optional[int] = None) -> None:
        self.json_path = json_path
        self.cfg_rate = float(cfg_rate)
        self.dataset_dir_path = os.path.dirname(json_path)
        self.prompt_embed_dir = os.path.join(self.dataset_dir_path, "prompt_embed")
        self.text_ids_dir = os.path.join(self.dataset_dir_path, "text_ids")
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        if num_sample:
            self.data_anno = self.data_anno[:num_sample]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.data_anno[idx]
        prompt_embed_file = item["prompt_embed_path"]
        text_ids_file = item["text_ids"]
        prompt_embeds = torch.load(
            os.path.join(self.prompt_embed_dir, prompt_embed_file),
            map_location="cpu",
            weights_only=True,
        )
        text_ids = torch.load(
            os.path.join(self.text_ids_dir, text_ids_file),
            map_location="cpu",
            weights_only=True,
        )
        caption = _extract_caption(item)
        if random.random() < self.cfg_rate:
            prompt_embeds = torch.zeros_like(prompt_embeds)
        return prompt_embeds, text_ids, caption

    def __len__(self) -> int:
        return len(self.data_anno)


def latent_collate_function(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    prompt_embeds, text_ids, captions = zip(*batch)
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)
    return prompt_embeds, text_ids, list(captions)


if __name__ == "__main__":
    dataset = LatentDataset("data/rl_embeddings/videos2caption.json", cfg_rate=0.0)
    sample = [dataset[0]]
    _ = latent_collate_function(sample)
