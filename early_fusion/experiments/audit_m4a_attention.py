"""Audit lanjutan M4a:
- Check A: Attention weights aktual (CLS attends to X%).
- Check B: Gradient ratio (image vs text vs tabular).
- Check C: Scale diagnostic (attention saat tabular × 0.1).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.environ["SEED"] = "42"
import torch
import torch.nn as nn

from early_fusion.models.ratf_pooled import RATF_M4a


def prepare_sequence(model, img, txt, mask, cont, gi):
    """Rebuild input sequence untuk forward manual. Return x, pad_mask, labels."""
    B, dev = cont.size(0), cont.device
    tab = model.tabular_tokenizer(cont, gi) + model.modality_emb.weight[2]

    parts = [model.cls_token.expand(B, -1, -1)]
    masks = [torch.zeros(B, 1, dtype=torch.bool, device=dev)]
    labels = ["CLS"]

    if model.use_text:
        eos_idx = (mask.sum(dim=1) - 1).long().clamp(min=0)
        txt_pooled = txt[torch.arange(B, device=dev), eos_idx]
        txt_tok = model.text_proj(model.text_ln(txt_pooled)).unsqueeze(1) + model.modality_emb.weight[0]
        parts.append(txt_tok)
        masks.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))
        labels.append("text")

    if model.use_image:
        img_pooled = img[:, 0]
        img_tok = model.image_proj(model.image_ln(img_pooled)).unsqueeze(1) + model.modality_emb.weight[1]
        parts.append(img_tok)
        masks.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))
        labels.append("image")

    parts.append(tab)
    masks.append(torch.zeros(B, tab.size(1), dtype=torch.bool, device=dev))
    labels.extend([f"tab_{i}" for i in range(tab.size(1))])

    x = torch.cat(parts, 1)
    pad_mask = torch.cat(masks, 1)
    return x, pad_mask, labels


def extract_attention(model, x, pad_mask):
    """Reimplement TransformerEncoder forward untuk capture attention weights per layer."""
    weights_per_layer = []
    for layer in model.transformer.encoder.layers:
        normed = layer.norm1(x)
        attn_out, attn_w = layer.self_attn(
            normed, normed, normed,
            key_padding_mask=pad_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        weights_per_layer.append(attn_w.detach())   # (B, nhead, L, L)
        x = x + attn_out
        x = x + layer._ff_block(layer.norm2(x))
    return weights_per_layer, x


def report_attention(weights, labels, title):
    print(f"--- {title} ---")
    for layer_i, w in enumerate(weights):
        # w: (B, nhead, L, L); CLS = index 0 di target
        cls_avg = w[:, :, 0, :].mean(dim=(0, 1))   # rata-rata batch + head → (L,)
        print(f"  Layer {layer_i+1}:")
        print(f"    CLS:     {cls_avg[0].item()*100:6.2f}%")
        idx = 1
        for name in labels[1:]:
            if name == "text":
                print(f"    text:    {cls_avg[idx].item()*100:6.2f}%")
                idx += 1
            elif name == "image":
                print(f"    image:   {cls_avg[idx].item()*100:6.2f}%")
                idx += 1
            else:
                break
        # Tabular: sum semua token tab_*
        tab_sum = cls_avg[idx:].sum().item() * 100
        print(f"    tabular: {tab_sum:6.2f}%   (sum over {cls_avg.shape[0]-idx} tokens)")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = RATF_M4a(
        image_dim=768, text_dim=512,
        n_continuous=11, n_genres=14,
        d=128, nhead=4, num_layers=2, dim_ff=512,
        dropout=0.2, embedding_noise_std=0.02,
        use_image=True, use_text=True,
    ).to(device)

    # Batch
    B = 32
    img = torch.randn(B, 50, 768, device=device)
    txt = torch.randn(B, 32, 512, device=device)
    mask = torch.ones(B, 32, dtype=torch.bool, device=device)
    cont = torch.randn(B, 11, device=device)
    gi = torch.zeros(B, dtype=torch.long, device=device)

    # ================= CHECK A: Attention Weights =================
    print("=" * 70)
    print("CHECK A: CLS attention distribution (train=eval, no dropout)")
    print("=" * 70)
    model.eval()
    with torch.no_grad():
        x, pad_mask, labels = prepare_sequence(model, img, txt, mask, cont, gi)
        print(f"Sequence length: {len(labels)}  labels: {labels}")
        print()
        weights, _ = extract_attention(model, x, pad_mask)
        report_attention(weights, labels, "Attention (original)")

    # ================= CHECK B: Gradient Ratio =================
    print("=" * 70)
    print("CHECK B: Gradient ratio per modality")
    print("=" * 70)
    model.train()
    y = torch.randn(B, device=device)
    out = model(img, txt, mask, cont, gi)
    loss = nn.HuberLoss()(out, y)
    model.zero_grad()
    loss.backward()

    def avg_grad(p):
        return p.grad.abs().mean().item() if p.grad is not None else 0.0

    g_img = avg_grad(model.image_proj.weight) + model.modality_emb.weight.grad[1].abs().mean().item()
    g_txt = avg_grad(model.text_proj.weight) + model.modality_emb.weight.grad[0].abs().mean().item()
    g_tab = avg_grad(model.tabular_tokenizer.weight) + model.modality_emb.weight.grad[2].abs().mean().item()
    total = g_img + g_txt + g_tab

    print(f"  Image:   {g_img:.6f}  → {g_img/total*100:6.2f}%")
    print(f"  Text:    {g_txt:.6f}  → {g_txt/total*100:6.2f}%")
    print(f"  Tabular: {g_tab:.6f}  → {g_tab/total*100:6.2f}%")
    print()

    # ================= CHECK C: Scale Diagnostic =================
    print("=" * 70)
    print("CHECK C: Attention under tabular scaled × 0.1 (diagnostic only)")
    print("=" * 70)
    model.eval()
    with torch.no_grad():
        x, pad_mask, labels = prepare_sequence(model, img, txt, mask, cont, gi)
        # Scale tabular token (posisi terakhir 12 token)
        n_tab = 12
        x_scaled = x.clone()
        x_scaled[:, -n_tab:] = x_scaled[:, -n_tab:] * 0.1
        weights_scaled, _ = extract_attention(model, x_scaled, pad_mask)
        report_attention(weights_scaled, labels, "Attention (tabular × 0.1)")


if __name__ == "__main__":
    main()