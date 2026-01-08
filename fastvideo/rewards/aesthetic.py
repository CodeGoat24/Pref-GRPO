from __future__ import annotations

from typing import List

import torch
from PIL import Image

from fastvideo.rewards.reward_paths import REWARD_MODEL_PATHS


class AestheticMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        checkpoint_path: str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the aesthetic reward."
            ) from exc
        self.clip = CLIPModel.from_pretrained(REWARD_MODEL_PATHS["aesthetic_clip"])
        self.processor = CLIPProcessor.from_pretrained(
            REWARD_MODEL_PATHS["aesthetic_clip"]
        )
        self.mlp = AestheticMLP()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.device = device
        self.clip.to(device=device, dtype=dtype)
        self.mlp.to(device=device, dtype=dtype)
        self.eval()

    @torch.no_grad()
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
