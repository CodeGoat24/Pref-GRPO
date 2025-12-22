import torch
from torch.nn import functional as F


def init_clip_model(
    device,
    model_name: str = "ViT-H-14",
    pretrained_path: str = "./open_clip_pytorch_model.bin",
):
    import open_clip

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_path,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model = clip_model.to(device)
    clip_model.eval()
    return clip_model, preprocess, tokenizer


def compute_clip_score(
    clip_model,
    preprocess,
    tokenizer,
    image,
    caption: str,
    device,
):
    with torch.no_grad():
        text = tokenizer([caption]).to(device=device, non_blocking=True)
        clip_image = preprocess(image).unsqueeze(0).to(device=device, non_blocking=True)
        clip_image_features = clip_model.encode_image(clip_image)
        clip_text_features = clip_model.encode_text(text)
        clip_image_features = F.normalize(clip_image_features, dim=-1)
        clip_text_features = F.normalize(clip_text_features, dim=-1)
        return (clip_image_features @ clip_text_features.T)[0]
