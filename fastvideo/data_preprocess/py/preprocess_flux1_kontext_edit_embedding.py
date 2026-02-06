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
import glob
import json
import os
from typing import Iterable, List

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from diffusers import FluxKontextPipeline
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

logger = get_logger(__name__)


def _iter_jsonl_paths(path_or_dir: str) -> List[str]:
    if os.path.isdir(path_or_dir):
        paths = sorted(glob.glob(os.path.join(path_or_dir, "*.jsonl")))
        if not paths:
            raise FileNotFoundError(f"No jsonl files found in {path_or_dir}")
        return paths
    if os.path.isfile(path_or_dir):
        return [path_or_dir]
    raise FileNotFoundError(f"Input path not found: {path_or_dir}")


def _load_jsonl(paths: Iterable[str]) -> List[dict]:
    items: List[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    return items


def _extract_instruction(item: dict, use_cn: bool) -> str:
    if use_cn and item.get("instruction_cn"):
        return str(item["instruction_cn"])
    for key in ("instruction", "prompt", "caption", "text"):
        if key in item and item[key] is not None:
            return str(item[key])
    return ""


def _coerce_items(batch_items) -> List[dict]:
    if isinstance(batch_items, list):
        return batch_items
    if isinstance(batch_items, dict):
        keys = list(batch_items.keys())
        if not keys:
            return []
        batch_size = len(batch_items[keys[0]])
        return [{k: batch_items[k][i] for k in keys} for i in range(batch_size)]
    raise TypeError(f"Unsupported item batch type: {type(batch_items)}")


class EditPromptDataset(Dataset):
    def __init__(self, items: List[dict], use_cn: bool) -> None:
        self.items = items
        self.use_cn = use_cn

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        instruction = _extract_instruction(item, self.use_cn)
        return {
            "instruction": instruction,
            "item": item,
            "filename": str(idx),
        }

    def __len__(self) -> int:
        return len(self.items)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    input_paths = _iter_jsonl_paths(args.input_path)
    if os.path.isdir(args.input_path):
        dataset_root = os.path.abspath(args.input_path)
    else:
        dataset_root = os.path.abspath(os.path.dirname(args.input_path))
    items = _load_jsonl(input_paths)
    if not items:
        raise ValueError("No edit items loaded from jsonl.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pooled_prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "text_ids"), exist_ok=True)

    dataset = EditPromptDataset(items, args.use_instruction_cn)
    sampler = DistributedSampler(dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    ).to(device)

    if local_rank == 0:
        with torch.inference_mode():
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = pipe.encode_prompt(
                prompt="",
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
            )
        torch.save(
            negative_prompt_embeds.cpu(),
            os.path.join(args.output_dir, "negative_prompt_embed.pt"),
        )
        torch.save(
            negative_pooled_prompt_embeds.cpu(),
            os.path.join(args.output_dir, "negative_pooled_prompt_embed.pt"),
        )
        torch.save(
            negative_text_ids.cpu(),
            os.path.join(args.output_dir, "negative_text_ids.pt"),
        )
    dist.barrier()

    json_data = []
    for _, data in tqdm(enumerate(dataloader), disable=local_rank != 0):
        try:
            with torch.inference_mode():
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=data["instruction"],
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                )
            items = _coerce_items(data["item"])
            for idx, sample_name in enumerate(data["filename"]):
                prompt_embed_path = os.path.join(args.output_dir, "prompt_embed", sample_name + ".pt")
                pooled_prompt_embed_path = os.path.join(
                    args.output_dir, "pooled_prompt_embed", sample_name + ".pt"
                )
                text_ids_path = os.path.join(args.output_dir, "text_ids", sample_name + ".pt")
                torch.save(prompt_embeds[idx].cpu(), prompt_embed_path)
                torch.save(pooled_prompt_embeds[idx].cpu(), pooled_prompt_embed_path)
                # Kontext text_ids are prompt-position ids and shared across batch.
                torch.save(text_ids.cpu(), text_ids_path)
                item = dict(items[idx])
                instruction = _extract_instruction(item, args.use_instruction_cn)
                item.update(
                    {
                        "instruction": instruction,
                        "prompt_embed_path": sample_name + ".pt",
                        "pooled_prompt_embed_path": sample_name + ".pt",
                        "text_ids": sample_name + ".pt",
                        "dataset_root": dataset_root,
                    }
                )
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
        output_json = args.output_json_path or os.path.join(args.output_dir, "edit_data.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_json_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="black-forest-labs/FLUX.1-Kontext-dev",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a jsonl file or a directory of jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where prompt embeddings and json metadata will be written.",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default=None,
        help="Optional output JSON path for merged metadata.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading. 0 means data is loaded in the main process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device for preprocessing.",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Max sequence length for Kontext text encoder.",
    )
    parser.add_argument(
        "--use_instruction_cn",
        action="store_true",
        help="Use instruction_cn when available instead of instruction.",
    )
    args = parser.parse_args()
    main(args)
