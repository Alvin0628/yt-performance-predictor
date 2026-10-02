"""Audit M4a: cek gradient flow & weight magnitude di jalur image/text/tabular."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.environ["SEED"] = "42"
import torch
import torch.nn as nn

# Import model dari ratf_pooled
from early_fusion.models.ratf_pooled import RATF_M4a


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Model full (use_image=True, use_text=True)
    model = RATF_M4a(
        image_dim=768, text_dim=512,
        n_continuous=11, n_genres=14,
        d=128, nhead=4, num_layers=2, dim_ff=512,
        dropout=0.2, embedding_noise_std=0.02,
        use_image=True, use_text=True,
    ).to(device)

    # Print init weights
    print()
    print("=== Init weights (setelah inisialisasi) ===")
    print(f"image_proj.weight: mean={model.image_proj.weight.abs().mean():.4f}, std={model.image_proj.weight.std():.4f}")
    print(f"image_proj.bias:   mean={model.image_proj.bias.abs().mean():.4f}")
    print(f"text_proj.weight:  mean={model.text_proj.weight.abs().mean():.4f}, std={model.text_proj.weight.std():.4f}")
    print(f"text_proj.bias:    mean={model.text_proj.bias.abs().mean():.4f}")
    print(f"image_ln.weight:   mean={model.image_ln.weight.abs().mean():.4f}")
    print(f"text_ln.weight:    mean={model.text_ln.weight.abs().mean():.4f}")
    print(f"modality_emb.weight:")
    for i, name in enumerate(["text", "image", "tabular"]):
        w = model.modality_emb.weight[i]
        print(f"  {name}: mean={w.abs().mean():.4f}, std={w.std():.4f}")
    print(f"tabular_tokenizer.weight: mean={model.tabular_tokenizer.weight.abs().mean():.4f}, std={model.tabular_tokenizer.weight.std():.4f}")
    print(f"tabular_tokenizer.bias:   mean={model.tabular_tokenizer.bias.abs().mean():.4f}")

    # Forward + backward
    print()
    print("=== Forward + backward (1 batch fake) ===")
    B = 8
    img = torch.randn(B, 50, 768, device=device)
    txt = torch.randn(B, 32, 512, device=device)
    mask = torch.ones(B, 32, dtype=torch.bool, device=device)
    cont = torch.randn(B, 11, device=device)
    gi = torch.zeros(B, dtype=torch.long, device=device)
    y = torch.randn(B, device=device)

    model.train()
    out = model(img, txt, mask, cont, gi)
    loss = nn.HuberLoss()(out, y)
    loss.backward()

    print()
    print("=== Gradient magnitudes (setelah 1 backward) ===")
    print(f"image_proj.weight.grad: mean={model.image_proj.weight.grad.abs().mean():.6f}")
    print(f"text_proj.weight.grad:  mean={model.text_proj.weight.grad.abs().mean():.6f}")
    print(f"tabular_tokenizer.weight.grad: mean={model.tabular_tokenizer.weight.grad.abs().mean():.6f}")
    print(f"modality_emb.weight.grad:")
    for i, name in enumerate(["text", "image", "tabular"]):
        g = model.modality_emb.weight.grad[i]
        print(f"  {name}: mean={g.abs().mean():.6f}")
    print(f"head[0].weight.grad:    mean={model.head[0].weight.grad.abs().mean():.6f}")

    # Cek LayerNorm stats dari image vs text
    print()
    print("=== LayerNorm output (input distributions) ===")
    with torch.no_grad():
        img_ln = model.image_ln(img[:, 0])  # CLS pooled
        txt_ln = model.text_ln(txt[:, -1])  # EOS pooled (fake)
        print(f"image_ln output: mean={img_ln.mean():.4f}, std={img_ln.std():.4f}")
        print(f"text_ln output:  mean={txt_ln.mean():.4f}, std={txt_ln.std():.4f}")

        img_proj = model.image_proj(img_ln)
        txt_proj = model.text_proj(txt_ln)
        print(f"image_proj output: mean={img_proj.mean():.4f}, std={img_proj.std():.4f}")
        print(f"text_proj output:  mean={txt_proj.mean():.4f}, std={txt_proj.std():.4f}")


        # Cek attention pattern
    print()
    print("=== Attention pattern (CLS attends to...) ===")
    model.eval()
    with torch.no_grad():
        # Hook ke transformer attention
        attn_weights = []
        def hook(module, input, output):
            # nn.TransformerEncoder menyimpan weights internal, sulit diambil
            # Alternatif: cek output embedding norm
            pass

        # Cara alternatif: forward dan cek per-token output norm di CLS position
        out = model(img, txt, mask, cont, gi)

        # Cek per-token norm sebelum final_ln (panggil transformer langsung)
        tab = model.tabular_tokenizer(cont, gi) + model.modality_emb.weight[2]
        parts = [model.cls_token.expand(B, -1, -1)]
        txt_p = model.text_proj(model.text_ln(txt[:, -1])).unsqueeze(1) + model.modality_emb.weight[0]
        img_p = model.image_proj(model.image_ln(img[:, 0])).unsqueeze(1) + model.modality_emb.weight[1]
        parts.append(txt_p)
        parts.append(img_p)
        parts.append(tab)
        x = torch.cat(parts, 1)
        zeros = torch.zeros(B, 1, dtype=torch.bool, device=device)
        zeros_tab = torch.zeros(B, 12, dtype=torch.bool, device=device)
        pad_mask = torch.cat([zeros, zeros, zeros, zeros_tab], dim=1)
        x_out = model.transformer(x, padding_mask=pad_mask)

        print(f"Token output norms (setelah transformer):")
        print(f"  CLS:     {x_out[:, 0].norm(dim=1).mean():.4f}")
        print(f"  text:    {x_out[:, 1].norm(dim=1).mean():.4f}")
        print(f"  image:   {x_out[:, 2].norm(dim=1).mean():.4f}")
        print(f"  tabular: {x_out[:, 3:].norm(dim=2).mean():.4f}")

if __name__ == "__main__":
    main()