import argparse
import os

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import pandas as pd

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

logger = get_logger(__name__)


class UniGenBenchDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.dataset = df["prompt_en"].tolist()
        self.index_list = df["index"].tolist()

    def __getitem__(self, idx):
        caption = self.dataset[idx]
        index = self.index_list[idx]
        return dict(caption=caption, idx=index)

    def __len__(self):
        return len(self.dataset)


def main(args):
    global_rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", global_rank))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "global rank", global_rank, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=global_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = UniGenBenchDataset(args.prompt_dir)
    sampler = DistributedSampler(
        dataset, rank=global_rank, num_replicas=world_size, shuffle=False
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
    if args.ckpt_path:
        if not os.path.isdir(args.ckpt_path):
            raise ValueError(f"Checkpoint path does not exist: {args.ckpt_path}")
        transformer = Flux2Transformer2DModel.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
        )
        pipe.transformer = transformer

    if args.lora_path:
        if not os.path.isdir(args.lora_path) and not os.path.isfile(args.lora_path):
            raise ValueError(f"LoRA path does not exist: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path, adapter_name="default")

    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=local_rank != 0)

    def build_prompt_kwargs(prompt, num_images):
        if not args.reuse_prompt_embeds or not hasattr(pipe, "encode_prompt"):
            return {"prompt": prompt}
        try:
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=tuple(args.text_encoder_out_layers),
            )
        except Exception:
            return {"prompt": prompt}
        return {"prompt_embeds": prompt_embeds}

    with torch.inference_mode():
        for _, data in tqdm(enumerate(dataloader), disable=local_rank != 0):
            try:
                captions = data["caption"]
                indices = data["idx"]
                for caption, idx in zip(captions, indices):
                    seeds = [args.seed + j for j in range(args.num_images_per_prompt)]
                    generators = [
                        torch.Generator(device=device).manual_seed(seed) for seed in seeds
                    ]
                    prompt_kwargs = build_prompt_kwargs(caption, args.num_images_per_prompt)
                    images = pipe(
                        height=args.height,
                        width=args.width,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        max_sequence_length=args.max_sequence_length,
                        text_encoder_out_layers=tuple(args.text_encoder_out_layers),
                        num_images_per_prompt=args.num_images_per_prompt,
                        generator=generators,
                        **prompt_kwargs,
                    ).images

                    for j, image in enumerate(images):
                        image_path = os.path.join(args.output_dir, f"{int(idx)}_{j}.png")
                        image.save(image_path)

            except Exception as exc:
                print(f"Rank {local_rank} Error: {repr(exc)}")
                dist.barrier()
                raise
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a transformer checkpoint directory (e.g. checkpoint-60-0).",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to a LoRA checkpoint directory or weights file.",
    )
    parser.add_argument("--prompt_dir", type=str, default="data/prompt_en.csv")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
    )
    parser.add_argument(
        "--reuse_prompt_embeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse text encodings if the pipeline supports encode_prompt.",
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model CPU offload to save GPU memory.",
    )
    args = parser.parse_args()
    main(args)
