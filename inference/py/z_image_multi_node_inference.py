import argparse
import inspect
import os

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from diffusers import DiffusionPipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel


class UniGenBenchDataset(Dataset):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.dataset = df["prompt_en"].tolist()
        if "index" in df.columns:
            self.index_list = df["index"].tolist()
        else:
            self.index_list = list(range(len(self.dataset)))

    def __getitem__(self, idx):
        return {"caption": self.dataset[idx], "idx": self.index_list[idx]}

    def __len__(self):
        return len(self.dataset)


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def init_distributed():
    global_rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", str(global_rank)))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=global_rank
        )
    return global_rank, local_rank, world_size


def main(args):
    global_rank, local_rank, world_size = init_distributed()
    print("world_size", world_size, "global rank", global_rank, "local rank", local_rank)

    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

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

    torch_dtype = parse_torch_dtype(args.torch_dtype)
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir,
    )

    if args.ckpt_path:
        if not os.path.isdir(args.ckpt_path):
            raise ValueError(f"Checkpoint path does not exist: {args.ckpt_path}")
        try:
            transformer = ZImageTransformer2DModel.from_pretrained(
                args.ckpt_path,
                torch_dtype=torch_dtype,
            )
        except Exception:
            transformer = ZImageTransformer2DModel.from_pretrained(
                args.ckpt_path,
                subfolder="transformer",
                torch_dtype=torch_dtype,
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

    call_sig = inspect.signature(pipe.__call__)
    call_params = set(call_sig.parameters.keys())

    with torch.inference_mode():
        for _, data in tqdm(enumerate(dataloader), disable=local_rank != 0):
            captions = data["caption"]
            indices = data["idx"]
            for caption, idx in zip(captions, indices):
                image_paths = [
                    os.path.join(args.output_dir, f"{int(idx)}_{j}.png")
                    for j in range(args.num_images_per_prompt)
                ]

                if args.skip_existing:
                    missing = [
                        (j, path)
                        for j, path in enumerate(image_paths)
                        if not os.path.exists(path)
                    ]
                else:
                    missing = list(enumerate(image_paths))
                if not missing:
                    continue

                seeds = [args.seed + j for j, _ in missing]
                generators = [
                    torch.Generator(device=device).manual_seed(seed)
                    for seed in seeds
                ]

                call_kwargs = {
                    "prompt": caption,
                    "height": args.height,
                    "width": args.width,
                    "num_inference_steps": args.num_inference_steps,
                    "num_images_per_prompt": len(missing),
                    "generator": generators,
                }
                if "guidance_scale" in call_params:
                    call_kwargs["guidance_scale"] = args.guidance_scale
                if "max_sequence_length" in call_params:
                    call_kwargs["max_sequence_length"] = args.max_sequence_length
                if "negative_prompt" in call_params and args.negative_prompt is not None:
                    call_kwargs["negative_prompt"] = args.negative_prompt
                if "cfg_normalization" in call_params:
                    call_kwargs["cfg_normalization"] = args.cfg_normalization
                if "cfg_truncation" in call_params:
                    call_kwargs["cfg_truncation"] = args.cfg_truncation

                images = pipe(**call_kwargs).images
                for image, (_, image_path) in zip(images, missing):
                    image.save(image_path)

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--prompt_dir", type=str, default="data/unigenbench_test_data.csv")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cfg_normalization", action="store_true")
    parser.add_argument("--cfg_truncation", type=float, default=1.0)
    parser.add_argument("--torch_dtype", type=str, default="bf16")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--enable_tf32", action="store_true")
    parser.add_argument("--enable_cpu_offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
