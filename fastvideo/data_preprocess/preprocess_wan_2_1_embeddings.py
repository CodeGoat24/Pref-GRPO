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
import torch
from accelerate.logging import get_logger
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline
from diffusers.utils import export_to_video
import json
import os
import torch.distributed as dist

logger = get_logger(__name__)
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from fastvideo.utils.load import load_text_encoder, load_vae
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm
import re
from diffusers import AutoencoderKLWan, WanPipeline

def contains_chinese(text):
    """检查字符串是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

class T5dataset(Dataset):
    def __init__(
        self, txt_path, vae_debug,
    ):
        self.txt_path = txt_path
        self.vae_debug = vae_debug
        with open(self.txt_path, "r", encoding="utf-8") as f:
            self.train_dataset = [
        line for line in f.read().splitlines() if not contains_chinese(line)
        ]
            #self.train_dataset = sorted(train_dataset)  

    def __getitem__(self, idx):
        #import pdb;pdb.set_trace()
        caption = self.train_dataset[idx]
        filename = str(idx)
        #length = self.train_dataset[idx]["length"]
        if self.vae_debug:
            latents = torch.load(
                os.path.join(
                    args.output_dir, "latent", self.train_dataset[idx]["latent_path"]
                ),
                map_location="cpu",
            )
        else:
            latents = []

        return dict(caption=caption, latents=latents, filename=filename)

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

    #videoprocessor = VideoProcessor(vae_scale_factor=8)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "negative_prompt_embeds"), exist_ok=True)

    latents_txt_path = args.prompt_dir
    train_dataset = T5dataset(latents_txt_path, args.vae_debug)
    pipe = WanPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)

    #vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    #vae.enable_tiling()
    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=data["caption"], do_classifier_free_guidance=True
            )
            if args.vae_debug:
                latents = data["latents"]
            for idx, video_name in enumerate(data["filename"]):
                prompt_embed_path = os.path.join(
                    args.output_dir, "prompt_embed", video_name + ".pt"
                )
                negative_prompt_embeds_path = os.path.join(
                    args.output_dir, "negative_prompt_embeds", video_name + ".pt"
                )
                # save latent
                torch.save(prompt_embeds[idx], prompt_embed_path)
                torch.save(negative_prompt_embeds[idx], negative_prompt_embeds_path)
                #print(f"sample {video_name} saved")
                #if args.vae_debug:
                    #   export_to_video(video[idx], video_path, fps=fps)
                item = {}
                #item["length"] = int(data["length"][idx])
                #item["latent_path"] = video_name + ".pt"
                item["prompt_embed_path"] = video_name + ".pt"
                item["negative_prompt_embeds_path"] = video_name + ".pt"
                item["caption"] = data["caption"][idx]
                json_data.append(item)
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    # text encoder & vae & diffusion model
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--vae_debug", action="store_true")
    parser.add_argument("--prompt_dir", type=str, default="./empty.txt")
    args = parser.parse_args()
    main(args)