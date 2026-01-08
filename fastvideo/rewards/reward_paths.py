from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

REWARD_MODEL_PATHS = {
    "clip_pretrained": str(_REPO_ROOT / "open_clip_pytorch_model.bin"),
    "hpsv2_ckpt": str(_REPO_ROOT / "hps_ckpt" / "HPS_v2.1_compressed.pt"),
    "pickscore_processor": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "pickscore_model": "yuvalkirstain/PickScore_v1",
    "hpsv3_ckpt": str(_REPO_ROOT / "HPSv3" / "HPSv3.safetensors"),
    "videoalign_ckpt": str(_REPO_ROOT / "videoalign_ckpt"),
    "aesthetic_ckpt": str(_REPO_ROOT / "assets" / "sac+logos+ava1-l14-linearMSE.pth"),
    "aesthetic_clip": "openai/clip-vit-large-patch14",
}


def get_reward_model_path(key: str) -> str:
    try:
        return REWARD_MODEL_PATHS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown reward model path key: {key}") from exc
