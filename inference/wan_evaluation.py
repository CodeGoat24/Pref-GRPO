import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TRITON_LOG_LEVEL", "ERROR")

import sys
import torch
from datetime import timedelta
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
import torchvision.io 

def truncate_filename_component(name, max_bytes):
    if max_bytes <= 0:
        return ""
    encoded = name.encode("utf-8")
    if len(encoded) <= max_bytes:
        return name
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class TxtPromptDataset(Dataset):
    """
    Dataset class to load prompts from a simple .txt file (one prompt per line).
    The index_list is crucial as it represents the original index for file naming.
    """
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
        return dict(caption=caption, idx=index, raw_caption=self.prompts[idx]) # 增加 raw_caption 方便命名

    def __len__(self):
        return len(self.prompts)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if args.enable_sdpa:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(False)

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for this distributed script.")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist_timeout = timedelta(seconds=args.dist_timeout_sec)
        dist.init_process_group(
            backend="nccl", 
            init_method="env://", 
            world_size=world_size, 
            rank=local_rank,
            timeout=dist_timeout,
        )
    else:
        dist_timeout = timedelta(seconds=args.dist_timeout_sec)

    def barrier_with_timeout():
        dist.barrier()
    
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

    total_prompts = sampler.num_samples * world_size
    progress = tqdm(
        total=total_prompts,
        desc="Generating",
        unit="prompt",
        disable=local_rank != 0,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    model_id = args.model_path 
    
    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    vae_dtype = dtype_map[args.vae_dtype]

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=vae_dtype)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    
    if args.pretrained_model_name_or_path:
        # original_transformer = pipe.transformer
        
        print(f'load transformer from {args.pretrained_model_name_or_path}...')
        transformer = WanTransformer3DModel.from_pretrained(    
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype = torch.bfloat16
        ).to(device)
        
            
        pipe.transformer = transformer

    if args.lora_ckpt_path is not None:
        lora_config_path = os.path.join(args.lora_ckpt_path, "lora_config.json")
        if os.path.exists(lora_config_path):
            import json
            with open(lora_config_path, "r") as f:
                lora_cfg = json.load(f)
        else:
            lora_cfg = None

        if lora_cfg is not None and isinstance(lora_cfg, dict):
            lora_rank = lora_cfg["lora_params"]["lora_rank"]
            lora_alpha = lora_cfg["lora_params"]["lora_alpha"]
            target_modules = lora_cfg["lora_params"]["target_modules"]
        else:
            # fallback to the defaults used during training
            lora_rank = 128
            lora_alpha = 256
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=False,
            target_modules=target_modules,
        )
        pipe.transformer.add_adapter(lora_config)
        # adapter is named "default" by diffusers/peft when not specified
        lora_state_dict = pipe.lora_state_dict(args.lora_ckpt_path)
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
        print(f"Loaded LoRA checkpoint from {args.lora_ckpt_path}")
    
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.compile_transformer and hasattr(torch, "compile"):
        pipe.transformer = torch.compile(pipe.transformer, mode=args.compile_mode)
    if args.compile_vae and hasattr(torch, "compile"):
        pipe.vae = torch.compile(pipe.vae, mode=args.compile_mode)


    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    height = args.height
    width = args.width
    num_frames = args.num_frames
    fps = 16

    num_samples_per_prompt = args.num_samples_per_prompt 
    
    for _, data in enumerate(dataloader):
        try:

            prompts = list(data["caption"])
            raw_captions = list(data["raw_caption"])
            idxs = list(data["idx"])


            for sample_index in range(num_samples_per_prompt): 
                with torch.inference_mode():
                    batch_prompts = []
                    batch_raw_captions = []
                    batch_idxs = []
                    batch_generators = []
                    batch_video_paths = []

                    for prompt, raw_caption, idx in zip(prompts, raw_captions, idxs):
                        seed = args.base_seed + int(idx) * num_samples_per_prompt + sample_index
                        suffix = f"-{sample_index}.mp4"
                        prompt_for_name = prompt
                        if len((prompt_for_name + suffix).encode("utf-8")) > 255:
                            prompt_for_name = truncate_filename_component(
                                prompt_for_name, 255 - len(suffix.encode("utf-8"))
                            )
                        video_path = f"{args.output_dir}/{prompt_for_name}{suffix}"
                        if os.path.exists(video_path):
                            continue
                        batch_prompts.append(prompt)
                        batch_raw_captions.append(raw_caption)
                        batch_idxs.append(idx)
                        batch_generators.append(torch.Generator(device=device).manual_seed(seed))
                        batch_video_paths.append(video_path)

                    if not batch_prompts:
                        continue

                    outputs = pipe(
                        prompt=batch_prompts,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=batch_generators,
                    ).frames

                    for output, video_path, prompt, idx in zip(
                        outputs, batch_video_paths, batch_prompts, batch_idxs
                    ):
                        export_to_video(output, video_path, fps=fps)

            batch_prompt_count = torch.tensor(len(prompts), device=device)
            dist.all_reduce(batch_prompt_count, op=dist.ReduceOp.SUM)
            if local_rank == 0:
                progress.update(int(batch_prompt_count.item()))

        except Exception as e:
            print(f"Rank {local_rank} Error on index {data.get('idx', ['N/A'])[0]}: {repr(e)}")
            barrier_with_timeout()
            raise 
            
    barrier_with_timeout()
    if local_rank == 0:
        progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataloader_num_workers", type=int, default=8, help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (per device).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="wan21_evaluation_videos", help="The output directory.",
    )
    parser.add_argument(
        "--model_path", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers", help="The path or name of the Wan2.1 Diffusers model.",
    )
    parser.add_argument(
        "--prompt_dir", type=str, default="./prompts/all_dimension.txt",
        help="Path to the .txt file containing prompts (one per line).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Number of frames for the generated video.",
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Video height.",
    )
    parser.add_argument(
        "--width", type=int, default=832, help="Video width.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=5.0, help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--base_seed", type=int, default=3407, help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help="",
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory (lora-checkpoint-xxx) for inference.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of diffusion steps for video generation."
    )
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=5,
        help="Number of samples per prompt.",
    )
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="VAE dtype for inference.",
    )
    parser.add_argument(
        "--enable_tf32",
        action="store_true",
        help="Enable TF32 for faster matmul on supported GPUs.",
    )
    parser.add_argument(
        "--enable_sdpa",
        action="store_true",
        help="Prefer Flash/Mem-Efficient SDPA if available.",
    )
    parser.add_argument(
        "--compile_transformer",
        action="store_true",
        help="Use torch.compile for the transformer.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode; use max-autotune for highest speed with extra tuning time/logs.",
    )
    parser.add_argument(
        "--compile_vae",
        action="store_true",
        help="Use torch.compile for the VAE.",
    )
    parser.add_argument(
        "--dist_timeout_sec",
        type=int,
        default=1800000,
        help="Timeout in seconds for torch.distributed init/barrier.",
    )


    args = parser.parse_args()

    # 运行主函数
    main(args)
