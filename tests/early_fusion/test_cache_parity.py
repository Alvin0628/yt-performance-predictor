
"""python -m tests.early_fusion.test_cache_parity   (from root repo)"""
import numpy as np
import torch
from PIL import Image

from early_fusion.datasets.clip_tokens import load_clip, image_tokens, text_tokens
from serving.features import encode_image, encode_title_clip


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, encode_fn, transform, tok, cfg = load_clip(device)

    img = Image.open("tests/fixtures/sample_thumbnail.jpg").convert("RGB")
    ref = encode_image(img, encode_fn, transform, cfg, device)
    t = image_tokens(clip, transform(img).unsqueeze(0).to(device))
    print("image tokens:", tuple(t.shape))
    assert t.shape == (1, 50, 768), f"shape salah: {t.shape}"

    for name, tt in [("fp32", t), ("fp16 roundtrip", t.half().float())]:
        with torch.inference_mode():
            pooled = clip.visual_projection(clip.vision_model.post_layernorm(tt[:, 0]))
        c = cos(ref, pooled.cpu().numpy())
        print(f"  {name}: cosine vs v1 = {c:.6f}")
        assert c > 0.999, f"parity gagal: {c}"

    titles = [
        "I Tried This For 30 Days",
        "Why I Quit Everything And Moved To A Tiny Island With No Internet For A Whole Year",
        "a very long title " * 8,
    ]
    for title in titles:
        ref = encode_title_clip(title, clip, cfg["hf_name"], device)
        h, m, ids = text_tokens(clip, tok, [title], device, max_tokens=77)
        eos = (ids == tok.eos_token_id).int().argmax(-1)
        with torch.inference_mode():
            pooled = clip.text_projection(h[torch.arange(1), eos])
        n_tok = int(m.sum())
        c = cos(ref, pooled.cpu().numpy())
        print(f"  {n_tok:2d} token, cosine = {c:.6f}  | {title[:40]}")
        assert c > 0.999, f"text parity gagal: {c}"

        h32, *_ = text_tokens(clip, tok, [title], device, max_tokens=32)
        k = min(n_tok - 1, 31)
        assert torch.allclose(h32[:, :k], h[:, :k], atol=1e-4), "kausalitas gagal"

    print("PARITY OK")


if __name__ == "__main__":
    main()