import argparse
import os
import torch
from accelerate.logging import get_logger
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video, convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict

logger = get_logger(__name__)


class TxtPromptDataset(Dataset):
    def __init__(self, txt_path):
        self.txt_path = txt_path

        try:
            with open(self.txt_path, "r", encoding="utf-8") as f:
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


def _extract_lora_params(lora_cfg, component_name):
    if not isinstance(lora_cfg, dict):
        return None
    lora_params = lora_cfg.get("lora_params")
    if not isinstance(lora_params, dict):
        return None
    if component_name in lora_params and isinstance(lora_params[component_name], dict):
        return lora_params[component_name]
    if "lora_rank" in lora_params:
        return lora_params
    return None


def _load_lora_component(
    component_name,
    model,
    lora_state_dict,
    lora_params,
    default_target_modules,
):
    if model is None:
        return False
    state_dict = {
        k.replace(f"{component_name}.", ""): v
        for k, v in lora_state_dict.items()
        if k.startswith(f"{component_name}.")
    }
    if not state_dict:
        return False
    if not isinstance(lora_params, dict):
        lora_params = {}
    lora_rank = lora_params.get("lora_rank", 128)
    lora_alpha = lora_params.get("lora_alpha", 256)
    target_modules = lora_params.get("target_modules", default_target_modules)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=False,
        target_modules=target_modules,
    )
    model.add_adapter(lora_config)
    state_dict = convert_unet_state_dict_to_peft(state_dict)
    set_peft_model_state_dict(model, state_dict, adapter_name="default")
    model.set_adapter("default")
    return True


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
            rank=local_rank,
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

    if args.lora_ckpt_path is not None:
        lora_config_path = os.path.join(args.lora_ckpt_path, "lora_config.json")
        if os.path.exists(lora_config_path):
            import json
            with open(lora_config_path, "r") as f:
                lora_cfg = json.load(f)
        else:
            lora_cfg = None

        lora_state_dict = pipe.lora_state_dict(args.lora_ckpt_path)
        default_target_modules = [
            "add_k_proj",
            "add_q_proj",
            "add_v_proj",
            "to_add_out",
            "to_k",
            "to_out.0",
            "to_q",
            "to_v",
        ]
        target_components = set()
        if isinstance(lora_cfg, dict):
            components = lora_cfg.get("target_components")
            if isinstance(components, (list, tuple)):
                target_components = set(components)

        loaded_any = False
        transformer_params = _extract_lora_params(lora_cfg, "transformer")
        if not target_components or "transformer" in target_components:
            loaded_any |= _load_lora_component(
                "transformer",
                pipe.transformer,
                lora_state_dict,
                transformer_params,
                default_target_modules,
            )

        transformer_2_params = _extract_lora_params(lora_cfg, "transformer_2")
        if transformer_2_params is None:
            transformer_2_params = transformer_params
        if not target_components or "transformer_2" in target_components:
            loaded_any |= _load_lora_component(
                "transformer_2",
                getattr(pipe, "transformer_2", None),
                lora_state_dict,
                transformer_2_params,
                default_target_modules,
            )

        if loaded_any:
            print(f"Loaded LoRA checkpoint from {args.lora_ckpt_path}")
        else:
            print(f"Warning: no matching LoRA weights found in {args.lora_ckpt_path}")

    pipe.to(device)

    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
        "in the background, walking backwards"
    )

    height = args.height
    width = args.width
    num_frames = args.num_frames
    fps = 15
    guidance_scale_2 = args.guidance_scale_2
    if guidance_scale_2 is not None and guidance_scale_2 < 0:
        guidance_scale_2 = None

    for _, data in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Rank {local_rank} Generating Videos",
        disable=local_rank != 0,
    ):
        try:
            for j in range(1):
                with torch.inference_mode():
                    seed = args.base_seed + j

                    prompt = data["caption"][0]
                    idx = data["idx"][0]

                    print(
                        f"Rank {local_rank} generating index {idx} (seed {seed}): '{prompt[:40]}...'"
                    )

                    call_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                        "guidance_scale": args.guidance_scale,
                        "generator": torch.Generator(device=device).manual_seed(seed),
                    }
                    if guidance_scale_2 is not None:
                        call_kwargs["guidance_scale_2"] = guidance_scale_2

                    output = pipe(**call_kwargs).frames[0]

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
        default="wan22_generated_videos",
        help="The output directory where the video predictions will be written.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="The path or name of the Wan2.2 Diffusers model.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="data/prompts.txt",
        help="Path to the .txt file containing prompts (one per line).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames for the generated video.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale for the pipeline.",
    )
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=3.0,
        help="Low-noise stage guidance scale for Wan2.2; set <0 to reuse --guidance_scale.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=3407,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory (lora-checkpoint-xxx) for inference.",
    )

    args = parser.parse_args()

    main(args)
