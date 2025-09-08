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
from diffusers import FluxPipeline
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import FluxPipeline
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    T5EncoderModel, T5TokenizerFast,
    CLIPVisionModelWithProjection,
)
import pandas as pd
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

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
    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)

    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder_2 = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)
    tokenizer_2 = T5TokenizerFast.from_pretrained(args.model_path, subfolder="tokenizer_2")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")

    pipe = FluxPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    for _, data in tqdm(enumerate(dataloader), disable=local_rank != 0):
        try:
            for j in range(4):
                with torch.inference_mode():
                    seed = 3407+j
                    prompt = data['caption'][0]
                    idx = data['idx'][0]
                    image = pipe(
                        prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=30,
                        max_sequence_length=512,
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

    parser.add_argument("--prompt_dir", type=str, default="data/prompt_en.csv")
    args = parser.parse_args()

    main(args)
