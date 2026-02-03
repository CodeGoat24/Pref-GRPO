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
import re

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from diffusers import Flux2KleinPipeline

logger = get_logger(__name__)


def contains_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


class PromptDataset(Dataset):
    def __init__(self, txt_path):
        self.txt_path = txt_path
        with open(self.txt_path, "r", encoding="utf-8") as f:
            self.train_dataset = [
                line for line in f.read().splitlines() if not contains_chinese(line)
            ]
        self.train_dataset = list(set(self.train_dataset))

    def __getitem__(self, idx):
        caption = self.train_dataset[idx]
        filename = str(idx)
        return dict(caption=caption, filename=filename)

    def __len__(self):
        return len(self.train_dataset)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "text_ids"), exist_ok=True)

    train_dataset = PromptDataset(args.prompt_dir)
    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    ).to(device)
    if local_rank == 0:
        with torch.inference_mode():
            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                prompt="",
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=tuple(args.text_encoder_out_layers),
            )
        torch.save(
            negative_prompt_embeds.cpu(),
            os.path.join(args.output_dir, "negative_prompt_embed.pt"),
        )
        torch.save(
            negative_text_ids.cpu(),
            os.path.join(args.output_dir, "negative_text_ids.pt"),
        )
    dist.barrier()

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        try:
            with torch.inference_mode():
                prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=data["caption"],
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                    text_encoder_out_layers=tuple(args.text_encoder_out_layers),
                )

            for idx, sample_name in enumerate(data["filename"]):
                prompt_embed_path = os.path.join(
                    args.output_dir, "prompt_embed", sample_name + ".pt"
                )
                text_ids_path = os.path.join(
                    args.output_dir, "text_ids", sample_name + ".pt"
                )
                torch.save(prompt_embeds[idx].cpu(), prompt_embed_path)
                torch.save(text_ids[idx].cpu(), text_ids_path)
                item = {
                    "prompt_embed_path": sample_name + ".pt",
                    "text_ids": sample_name + ".pt",
                    "caption": data["caption"][idx],
                }
                json_data.append(item)
        except Exception as exc:
            print(f"Rank {local_rank} Error: {repr(exc)}")
            dist.barrier()
            raise

    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4B",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading. 0 means data is loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the preprocessing dataloader.",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where prompt embeddings and json metadata will be written.",
    )
    parser.add_argument("--prompt_dir", type=str, default="./empty.txt")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Max sequence length for the Qwen3 text encoder.",
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
        help="Hidden layers to extract for Qwen3 prompt embeddings.",
    )
    args = parser.parse_args()
    main(args)
