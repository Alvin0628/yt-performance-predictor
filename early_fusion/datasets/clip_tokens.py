"""Ekstraktor token CLIP untuk RATF.

Preprocessing dan encoder di-import dari kode teman supaya satu sumber kebenaran.
"""
import torch
from transformers import AutoTokenizer

from models.precompute_embeddings import ENCODERS, build_transform, build_image_encoder

ENCODER_KEY = "clip_b32"
IMAGE_MODE = "squash"
MAX_TEXT_TOKENS = 32


def load_clip(device):
    cfg = ENCODERS[ENCODER_KEY]
    encode_fn, clip_model = build_image_encoder(cfg, device)
    transform = build_transform(cfg, IMAGE_MODE)
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    return clip_model, encode_fn, transform, tokenizer, cfg


@torch.inference_mode()
def image_tokens(clip_model, x):
    """x: (B,3,224,224). Return (B,50,768): [CLS] + 49 patch (pre-post_LN)."""
    return clip_model.vision_model(pixel_values=x).last_hidden_state


@torch.inference_mode()
def text_tokens(clip_model, tokenizer, titles, device, max_tokens=MAX_TEXT_TOKENS):
    """Return (B,L,512), (B,L) mask, (B,L) input_ids."""
    enc = tokenizer(
        titles,
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    ).to(device)
    hidden = clip_model.text_model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    ).last_hidden_state
    return hidden, enc["attention_mask"], enc["input_ids"]