"""Diagnostic: cek apakah last_hidden_state vision_model sudah post-LN atau belum."""
import torch
from transformers import CLIPModel
from models.precompute_embeddings import ENCODERS, build_transform
from PIL import Image


def main():
    cfg = ENCODERS["clip_b32"]
    model = CLIPModel.from_pretrained(cfg["hf_name"]).eval()
    transform = build_transform(cfg, "squash")

    img = Image.open("tests/fixtures/sample_thumbnail.jpg").convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.inference_mode():
        out = model.vision_model(pixel_values=x)
        last = out.last_hidden_state
        pool = out.pooler_output

        # Case A: last sudah post-LN → pool == last[:, 0]
        a = (pool - last[:, 0]).abs().max().item()

        # Case B: last masih pre-LN → pool == post_layernorm(last)[:, 0]
        b = (pool - model.vision_model.post_layernorm(last)[:, 0]).abs().max().item()

    print("last_hidden_state shape:", tuple(last.shape))
    print("pooler_output shape    :", tuple(pool.shape))
    print("Case A (last sudah post-LN):", a)
    print("Case B (last masih pre-LN) :", b)


if __name__ == "__main__":
    main()