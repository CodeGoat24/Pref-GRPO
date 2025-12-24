import argparse
import json
import torch
from accelerate.logging import get_logger
import os
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict

logger = get_logger(__name__)

class TxtPromptDataset(Dataset):
    def __init__(self, txt_path):
        self.txt_path = txt_path
        
        try:
            with open(self.txt_path, 'r', encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at: {txt_path}")
        except Exception as e:
            raise IOError(f"Error reading prompt file: {e}")
        
        self.index_list = list(range(len(self.prompts)))
    
    def __getitem__(self, idx):
        caption = self.prompts[idx]
        index = self.index_list[idx]
        return dict(caption=caption, idx=index)

    def __len__(self):
        return len(self.prompts)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print(f"World Size: {world_size}, Local Rank: {local_rank}")


    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for this distributed script.")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", 
            init_method="env://", 
            world_size=world_size, 
            rank=local_rank
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset = TxtPromptDataset(args.prompt_dir)
    
    sampler = DistributedSampler(
        dataset, rank=local_rank, num_replicas=world_size, shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    model_id = args.model_path 
    
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    
    pipe.to(device)
    if args.lora_dir:
        if not os.path.isdir(args.lora_dir):
            raise ValueError(f"LORA_DIR not found: {args.lora_dir}")
        lora_config_path = os.path.join(args.lora_dir, "lora_config.json")
        if os.path.exists(lora_config_path):
            with open(lora_config_path, "r") as f:
                lora_cfg = json.load(f)
        else:
            lora_cfg = None
        if lora_cfg is not None and isinstance(lora_cfg, dict):
            lora_rank = lora_cfg["lora_params"]["lora_rank"]
            lora_alpha = lora_cfg["lora_params"]["lora_alpha"]
            target_modules = lora_cfg["lora_params"]["target_modules"]
        else:
            lora_rank = 128
            lora_alpha = 256
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
                "to_k",
                "to_out.0",
                "to_q",
                "to_v",
            ]
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=False,
            target_modules=target_modules,
        )
        pipe.transformer.add_adapter(lora_config)
        lora_state_dict = pipe.lora_state_dict(args.lora_dir)
        transformer_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        set_peft_model_state_dict(
            pipe.transformer, transformer_state_dict, adapter_name="default"
        )
        pipe.transformer.set_adapter("default")
        print(f"Loaded LoRA checkpoint from {args.lora_dir}")

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    height = 480
    width = 832
    
    # 视频生成参数
    num_frames = args.num_frames
    fps = 15

    for _, data in tqdm(enumerate(dataloader), 
                        total=len(dataloader), 
                        desc=f"Rank {local_rank} Generating Videos",
                        disable=local_rank != 0):
        try:

            for j in range(1): 
                with torch.inference_mode():
                    seed = args.base_seed + j
                    
                    prompt = data['caption'][0]
                    idx = data['idx'][0]
                    
                    print(f"Rank {local_rank} generating index {idx} (seed {seed}): '{prompt[:40]}...'")

                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        guidance_scale=args.guidance_scale, 
                        generator=torch.Generator(device=device).manual_seed(seed)
                    ).frames[0]

                    video_path = f"{args.output_dir}/{str(int(idx))}_{j}.mp4"
                    export_to_video(output, video_path, fps=fps)

        except Exception as e:
            print(f"Rank {local_rank} Error on index {data.get('idx', ['N/A'])[0]}: {repr(e)}")
            dist.barrier()
            raise 
            
    dist.barrier()
    if local_rank == 0:
        print("\nAll ranks finished video generation successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the dataloader. Recommended 1 for T2V inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="wan21_generated_videos",
        help="The output directory where the video predictions will be written.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="The path or name of the Wan2.1 Diffusers model.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Optional LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--prompt_dir", 
        type=str, 
        default="data/video_prompts.txt",
        help="Path to the .txt file containing prompts (one per line)."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames for the generated video."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale for the pipeline."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=3407,
        help="Base seed for reproducibility."
    )
                        
    args = parser.parse_args()

    main(args)
