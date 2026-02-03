import argparse
import torch
from accelerate.logging import get_logger
import os
import torch.distributed as dist

logger = get_logger(__name__)
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import json
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

class UniGenBenchDataset(Dataset):
    def __init__(
        self, csv_path,
    ):
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
    dataset = UniGenBenchDataset(args.prompt_dir)
    sampler = DistributedSampler(
        dataset, rank=local_rank, num_replicas=world_size, shuffle=False
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    from diffusers import DiffusionPipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if args.lora_dir:
        if not os.path.isdir(args.lora_dir):
            raise ValueError(f"LORA_DIR not found: {args.lora_dir}")
        lora_config_path = os.path.join(args.lora_dir, "lora_config.json")
        if os.path.exists(lora_config_path):
            with open(lora_config_path, "r") as f:
                lora_cfg = json.load(f)
        else:
            lora_cfg = None
        if lora_cfg is None or not isinstance(lora_cfg, dict):
            raise ValueError(f"Missing lora_config.json in {args.lora_dir}")
        lora_rank = lora_cfg["lora_params"]["lora_rank"]
        lora_alpha = lora_cfg["lora_params"]["lora_alpha"]
        target_modules = lora_cfg["lora_params"]["target_modules"]
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=False,
            target_modules=target_modules,
        )
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            raise ValueError("Pipeline has no transformer; cannot load LoRA.")
        if hasattr(transformer, "add_adapter"):
            transformer.add_adapter(lora_config)
        else:
            from peft import get_peft_model

            transformer = get_peft_model(transformer, lora_config)
            pipe.transformer = transformer
        candidate_files = [
            "adapter_model.safetensors",
            "pytorch_lora_weights.safetensors",
            "lora.safetensors",
        ]
        weight_path = None
        for fname in candidate_files:
            path = os.path.join(args.lora_dir, fname)
            if os.path.exists(path):
                weight_path = path
                break
        if weight_path is None:
            raise ValueError(f"No LoRA weights found in {args.lora_dir}")
        lora_state_dict = load_file(weight_path)
        if any(k.startswith("transformer.") for k in lora_state_dict.keys()):
            lora_state_dict = {
                k.replace("transformer.", ""): v for k, v in lora_state_dict.items()
            }
        if not any(k.startswith("lora") for k in lora_state_dict.keys()):
            lora_state_dict = convert_unet_state_dict_to_peft(lora_state_dict)
        set_peft_model_state_dict(
            transformer, lora_state_dict, adapter_name="default"
        )
        if hasattr(transformer, "set_adapter"):
            transformer.set_adapter("default")
        print(f"Loaded LoRA checkpoint from {args.lora_dir}")

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
        "zh": ", 超清，4K，电影级构图." # for chinese prompt
    }

    negative_prompt = " "

    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    width, height = aspect_ratios["16:9"]

    for _, data in tqdm(enumerate(dataloader), disable=local_rank != 0):
        try:
            for j in range(4):
                with torch.inference_mode():
                    seed = 3407+j
                    prompt = data['caption'][0]
                    idx = data['idx'][0]
                    image = pipe(
                        prompt=prompt + positive_magic["en"],
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=50,
                        true_cfg_scale=4.0,
                        generator=torch.Generator(device="cuda").manual_seed(seed)
                    ).images[0]

                    image_path = f"{args.output_dir}/{str(int(idx))}_{j}.png"
                    image.save(image_path)

        except Exception as e:
            print(f"Rank {local_rank} Error: {repr(e)}")
            dist.barrier()
            raise 
    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Optional LoRA checkpoint directory.",
    )

    parser.add_argument("--prompt_dir", type=str, default="data/prompt_en.csv")
    args = parser.parse_args()

    main(args)
