import os

from diffusers.utils import export_to_video


def _get_rollout_root(output_dir):
    return os.path.join(output_dir, "rollout") if output_dir else "rollout"


def save_rollout_image(image, output_dir, filename):
    rollout_root = _get_rollout_root(output_dir)
    os.makedirs(rollout_root, exist_ok=True)
    save_path = os.path.join(rollout_root, filename)
    image.save(save_path)
    return save_path


def save_rollout_video(video, output_dir, filename, fps=16):
    rollout_root = _get_rollout_root(output_dir)
    os.makedirs(rollout_root, exist_ok=True)
    save_path = os.path.join(rollout_root, filename)
    export_to_video(video, save_path, fps=fps)
    return save_path
