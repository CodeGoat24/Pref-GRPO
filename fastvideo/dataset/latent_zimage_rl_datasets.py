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

import torch
from torch.utils.data import Dataset
import json
import os
import random


class LatentDataset(Dataset):
    def __init__(
        self, json_path, num_latent_t, cfg_rate,
    ):
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.prompt_attention_mask_dir = os.path.join(
            self.datase_dir_path, "prompt_attention_mask"
        )
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        self.num_latent_t = num_latent_t

        uncond_embed_path = os.path.join(self.datase_dir_path, "uncond_prompt_embed.pt")
        uncond_mask_path = os.path.join(self.datase_dir_path, "uncond_prompt_attention_mask.pt")
        uncond_meta_path = os.path.join(self.datase_dir_path, "uncond_meta.json")

        if os.path.exists(uncond_embed_path) and os.path.exists(uncond_mask_path):
            self.uncond_prompt_embed = torch.load(uncond_embed_path, map_location="cpu", weights_only=True).to(
                torch.float32
            )
            self.uncond_prompt_mask = torch.load(uncond_mask_path, map_location="cpu", weights_only=True).to(torch.bool)
            self.uncond_original_length = int(self.uncond_prompt_mask.sum().item())
        else:
            # Fallback shape from the first training sample if unconditional cache is not present.
            first = self.data_anno[0]
            sample_embed = torch.load(
                os.path.join(self.prompt_embed_dir, first["prompt_embed_path"]),
                map_location="cpu",
                weights_only=True,
            ).to(torch.float32)
            self.uncond_prompt_embed = torch.zeros_like(sample_embed)
            self.uncond_prompt_mask = torch.zeros((sample_embed.shape[0],), dtype=torch.bool)
            self.uncond_original_length = 0

        if os.path.exists(uncond_meta_path):
            with open(uncond_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.uncond_original_length = int(meta.get("original_length", self.uncond_original_length))

        self.lengths = [
            data_item["length"] if "length" in data_item else 1
            for data_item in self.data_anno
        ]

    def __getitem__(self, idx):
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        prompt_attention_mask_file = self.data_anno[idx]["prompt_attention_mask"]
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
            prompt_attention_mask = self.uncond_prompt_mask
            original_length = self.uncond_original_length
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            ).to(torch.float32)
            prompt_attention_mask = torch.load(
                os.path.join(
                    self.prompt_attention_mask_dir, prompt_attention_mask_file
                ),
                map_location="cpu",
                weights_only=True,
            ).to(torch.bool)
            original_length = int(self.data_anno[idx].get("original_length", int(prompt_attention_mask.sum().item())))
        return prompt_embed, prompt_attention_mask, self.data_anno[idx]["caption"], original_length

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    prompt_embeds, prompt_attention_masks, caption, original_length = zip(*batch)
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    return prompt_embeds, prompt_attention_masks, caption, original_length
