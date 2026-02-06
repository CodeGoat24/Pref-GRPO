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
from typing import List, Tuple

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class PromptDataset(Dataset):
    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        with open(self.txt_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.read().splitlines() if line.strip()]

    def __getitem__(self, idx: int):
        return {"caption": self.prompts[idx], "filename": str(idx)}

    def __len__(self):
        return len(self.prompts)


def _init_distributed() -> Tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", str(rank)))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    should_init = world_size > 1 or ("MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ)
    if should_init and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def _pad_prompt_embed(prompt_embed: torch.Tensor, target_length: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq_len, hidden_dim = prompt_embed.shape
    original_length = min(seq_len, target_length)

    padded_embed = torch.zeros((target_length, hidden_dim), dtype=torch.float32)
    attention_mask = torch.zeros((target_length,), dtype=torch.bool)

    if original_length > 0:
        padded_embed[:original_length] = prompt_embed[:original_length].to(torch.float32).cpu()
        attention_mask[:original_length] = True

    return padded_embed, attention_mask, original_length


def main(args):
    rank, local_rank, world_size = _init_distributed()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_embed_dir = os.path.join(args.output_dir, "prompt_embed")
    prompt_attention_mask_dir = os.path.join(args.output_dir, "prompt_attention_mask")
    os.makedirs(prompt_embed_dir, exist_ok=True)
    os.makedirs(prompt_attention_mask_dir, exist_ok=True)

    dataset = PromptDataset(args.prompt_dir)
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    dtype = torch.bfloat16 if use_cuda else torch.float32
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    if not hasattr(pipe, "encode_prompt"):
        raise ValueError(f"Pipeline loaded from {args.model_path} does not provide encode_prompt().")

    # Prepare unconditional prompt embedding for CFG dropout in training.
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            uncond_embeds, _ = pipe.encode_prompt(
                prompt=[""],
                do_classifier_free_guidance=False,
                max_sequence_length=args.max_sequence_length,
            )
    uncond_padded, uncond_mask, uncond_len = _pad_prompt_embed(uncond_embeds[0], args.max_sequence_length)
    torch.save(uncond_padded, os.path.join(args.output_dir, "uncond_prompt_embed.pt"))
    torch.save(uncond_mask, os.path.join(args.output_dir, "uncond_prompt_attention_mask.pt"))
    with open(os.path.join(args.output_dir, "uncond_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"original_length": int(uncond_len)}, f, ensure_ascii=False, indent=2)

    local_json_data: List[dict] = []
    for _, batch in tqdm(enumerate(dataloader), disable=rank != 0):
        captions = list(batch["caption"])
        filenames = list(batch["filename"])

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                prompt_embeds_list, _ = pipe.encode_prompt(
                    prompt=captions,
                    do_classifier_free_guidance=False,
                    max_sequence_length=args.max_sequence_length,
                )

        for idx, filename in enumerate(filenames):
            prompt_embed, prompt_attention_mask, original_length = _pad_prompt_embed(
                prompt_embeds_list[idx], args.max_sequence_length
            )

            prompt_embed_file = f"{filename}.pt"
            prompt_mask_file = f"{filename}.pt"
            torch.save(prompt_embed, os.path.join(prompt_embed_dir, prompt_embed_file))
            torch.save(prompt_attention_mask, os.path.join(prompt_attention_mask_dir, prompt_mask_file))

            local_json_data.append(
                {
                    "prompt_embed_path": prompt_embed_file,
                    "prompt_attention_mask": prompt_mask_file,
                    "caption": captions[idx],
                    "original_length": int(original_length),
                }
            )

    if dist.is_initialized():
        dist.barrier()
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_json_data)
        all_json_data = [item for sublist in gathered_data for item in sublist]
    else:
        all_json_data = local_json_data

    if rank == 0:
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w", encoding="utf-8") as f:
            json.dump(all_json_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Z-Image model path or HF model id.")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Text file with one prompt per line.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for preprocessed embeddings.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Pad/truncate sequence length for prompt embeddings.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for preprocessing.",
    )
    main(parser.parse_args())
