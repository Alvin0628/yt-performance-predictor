"""M6 (granular) v2 training on canonical temporal_no_subs split.

Perubahan dari train_m6_temporal_granular.py (cari tanda [CHANGED] / [NEW]):
[NEW]     CLI: --gate-mode {softmax,sigmoid2}, --mod-dropout P, --gate-lr-mult X
          -> bisa ablasi 1 perubahan per run.
[CHANGED] Model: RATF_M6_Granular_V2.
[NEW]     Optimizer param group terpisah untuk gate: LR = LR * gate_lr_mult, weight_decay = 0.
[CHANGED] WARMUP_STEPS 300 -> 100 (epoch = 9028/64 = 142 step; 300 step ~ 2 epoch,
          padahal best epoch multimodal cuma 4-11).
[CHANGED] PATIENCE 10 -> 15 (modality dropout memperlambat konvergensi val loss).
[CHANGED] Statistik reliability dihitung di SELURUH val set (sebelumnya hanya batch pertama),
          dengan mean/std/min/max antar sample per modality.
[NEW]     Gate mean per epoch dicetak di baris log (cek apakah gate bergerak).
[CHANGED] Output: file results/checkpoint/variant memakai prefix m6v2 + tag konfigurasi,
          jadi tidak menimpa hasil lama.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_absolute_error

from models.dataset import build_tabular_matrix
from early_fusion.datasets.token_cache import load_cache
from early_fusion.datasets.token_dataset import TokenDataset
from early_fusion.models.ratf_m6_granular_v2 import RATF_M6_Granular_V2   # [CHANGED]
from early_fusion.experiments._common import (
    set_seed, hash_ids, get_git_info, load_snapshot,
)
from early_fusion.splits.load_split import load_canonical_split, apply_split_to_df


SEED = int(os.environ.get("SEED", 42))
SNAPSHOT_HASH = "c14dba895034fc4c"
CACHE_DIR = "data_snapshots/token_cache"
RESULTS = Path("early_fusion/results/m6v2_temporal_granular_results.jsonl")   # [CHANGED]
CHECKPOINT_DIR = Path("early_fusion/models/checkpoints")

D = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 512
DROPOUT = 0.2
EMB_NOISE = 0.02
LR = 1e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 15          # [CHANGED] was 10
WARMUP_STEPS = 100     # [CHANGED] was 300
GRAD_CLIP = 1.0
CROSS_ATTN_LAYERS = 1
N_IMAGE_TOKENS = 50
N_TEXT_TOKENS = 32
GATE_DROPOUT = 0.1     # [NEW]
SUBS_IDX = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    # [NEW] knob ablasi
    parser.add_argument("--gate-mode", choices=["softmax", "sigmoid2"], default="softmax")
    parser.add_argument("--mod-dropout", type=float, default=0.0)
    parser.add_argument("--gate-lr-mult", type=float, default=10.0)
    args = parser.parse_args()

    use_image = not args.no_image
    use_text = not args.no_text
    if not use_image and not use_text:
        base_variant = "m6v2_temporal_granular_tabular_only"
    elif not use_image:
        base_variant = "m6v2_temporal_granular_no_image"
    elif not use_text:
        base_variant = "m6v2_temporal_granular_no_text"
    else:
        base_variant = "m6v2_temporal_granular_full"
    # [NEW] tag konfigurasi supaya run ablasi tidak saling menimpa
    tag = f"{args.gate_mode}_md{args.mod_dropout:g}_glr{args.gate_lr_mult:g}"
    variant = f"{base_variant}__{tag}"

    set_seed(SEED)
    git_sha, git_dirty = get_git_info()
    print(f"git: {git_sha[:8]} dirty={git_dirty}")
    print(f"variant: {variant} (use_image={use_image}, use_text={use_text})")
    print(f"gate: mode={args.gate_mode} mod_dropout={args.mod_dropout} gate_lr_mult={args.gate_lr_mult}")
    print(f"split: canonical temporal_no_subs, drop_subs: True")
    print(f"cross_attn_layers: {CROSS_ATTN_LAYERS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df, _, _ = load_snapshot()

    split = load_canonical_split(verbose=True)
    train_idx, val_idx, test_idx = apply_split_to_df(df, split)
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    split_hashes = {
        "train_ids_hash": split["train_ids_hash"],
        "val_ids_hash": split["val_ids_hash"],
        "test_ids_hash": split["test_ids_hash"],
    }

    genres_train = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    n_genres_with_unk = len(genres_train) + 1

    _, scaler = build_tabular_matrix(
        df.iloc[train_idx], genres_train, fit_scaler=True)
    all_tab, _ = build_tabular_matrix(df, genres_train, scaler=scaler)
    all_tab = np.delete(all_tab, SUBS_IDX, axis=1)
    cont_all_full = all_tab.astype(np.float32)
    n_cont = cont_all_full.shape[1]
    print(f"n_cont={n_cont}, n_genres(+unk)={n_genres_with_unk} (drop subs from {n_cont + 1})")

    genre_to_idx = {g: i + 1 for i, g in enumerate(genres_train)}
    genre_idx = np.array(
        [genre_to_idx.get(g, 0) for g in df["genre"].fillna("")], dtype=np.int64
    )

    img_tokens, txt_tokens, txt_mask, thumb_ok, index_df, cache_meta = load_cache(CACHE_DIR)
    assert (index_df["video_id"].values == df["video_id"].values).all()
    assert cache_meta["snapshot_hash"] == SNAPSHOT_HASH

    targets = df["target"].values.astype(np.float32)
    video_ids = df["video_id"].values

    full_ds = TokenDataset(img_tokens, txt_tokens, txt_mask,
                           cont_all_full, genre_idx, targets, video_ids)
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=BATCH_SIZE, num_workers=0)

    # ---------- Model ----------
    model = RATF_M6_Granular_V2(                                   # [CHANGED]
        image_dim=768, text_dim=512,
        n_continuous=n_cont, n_genres=n_genres_with_unk,
        d=D, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
        n_image_tokens=N_IMAGE_TOKENS, n_text_tokens=N_TEXT_TOKENS,
        dropout=DROPOUT, embedding_noise_std=EMB_NOISE,
        use_image=use_image, use_text=use_text,
        cross_attn_layers=CROSS_ATTN_LAYERS,
        gate_mode=args.gate_mode,                                  # [NEW]
        gate_dropout=GATE_DROPOUT,                                 # [NEW]
        modality_dropout=args.mod_dropout,                         # [NEW]
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    # ---------- [CHANGED] Optimizer dengan param group untuk gate ----------
    gate_params = list(model.gate.parameters()) if model.gate is not None else []
    gate_ids = {id(p) for p in gate_params}
    base_params = [p for p in model.parameters() if id(p) not in gate_ids]
    groups = [{"params": base_params, "lr": LR, "weight_decay": WEIGHT_DECAY}]
    if gate_params:
        groups.append({"params": gate_params, "lr": LR * args.gate_lr_mult, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)   # lambda yang sama berlaku ke semua group
    loss_fn = nn.HuberLoss()

    def run_epoch(loader, train=False, collect_rel=False):
        model.train() if train else model.eval()
        total, n = 0.0, 0
        preds, ys = [], []
        gate_ws = []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                img = batch["image_tokens"].to(device)
                txt = batch["text_tokens"].to(device)
                mask = batch["text_mask"].to(device)
                tab = batch["tabular"].to(device)
                gi = batch["genre_idx"].to(device)
                y = batch["target"].float().to(device)

                if train:
                    opt.zero_grad()
                p = model(img, txt, mask, tab, gi, return_reliability=collect_rel)
                if collect_rel:
                    _, w = model.get_last_gate_weights()
                    if w is not None:
                        gate_ws.append(w.cpu())
                loss = loss_fn(p, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                    sched.step()
                total += loss.item() * len(y)
                n += len(y)
                preds.extend(p.detach().cpu().numpy().tolist())
                ys.extend(y.cpu().numpy().tolist())
        return total / n, np.array(preds), np.array(ys), gate_ws

    def summarize_gate(gate_ws):
        """[NEW] mean/std/min/max antar sample per modality, di seluruh set."""
        if not gate_ws:
            return {}
        W = torch.cat(gate_ws, dim=0)
        stats = {}
        for j, name in enumerate(model.mod_names):
            col = W[:, j]
            stats[f"{name}_mean"] = float(col.mean())
            stats[f"{name}_std"] = float(col.std())
            stats[f"{name}_min"] = float(col.min())
            stats[f"{name}_max"] = float(col.max())
        return stats

    best_val = float("inf"); best_state = None; best_epoch = 0; stale = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, _, _, _ = run_epoch(train_loader, train=True)
        va_loss, va_preds, va_targets, va_gate = run_epoch(val_loader, train=False, collect_rel=True)
        va_sp = spearmanr(va_preds, va_targets)[0]
        g = summarize_gate(va_gate)
        g_str = " ".join(f"{k.replace('_mean','')}={v:.3f}" for k, v in g.items() if k.endswith("_mean"))

        if va_loss < best_val:
            best_val = va_loss; best_epoch = epoch; stale = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"ep {epoch:3d} | tr={tr_loss:.4f} va={va_loss:.4f} sp={va_sp:.4f} | w[{g_str}] *")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop ep {epoch}")
                break

    model.load_state_dict(best_state)
    _, val_preds, val_targets, val_gate = run_epoch(val_loader, train=False, collect_rel=True)
    val_sp = spearmanr(val_preds, val_targets)[0]
    has_both = len(np.unique((val_targets > 0).astype(int))) == 2
    val_auc = roc_auc_score((val_targets > 0).astype(int), val_preds) if has_both else float("nan")

    rel_stats = summarize_gate(val_gate)   # [CHANGED] seluruh val set, bukan batch pertama

    print()
    print(f"=== M6v2 temporal granular val (seed={SEED}, variant={variant}) ===")
    print(f"val Spearman: {val_sp:.4f}")
    print(f"val AUC:      {val_auc:.4f}")
    print(f"best epoch:   {best_epoch}, params: {n_params:,}")
    print(f"gate weights (val, antar sample):")
    for k, v in rel_stats.items():
        print(f"  {k}: {v:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"m6v2_temporal_granular_seed{SEED}_{base_variant.split('granular_')[-1]}__{tag}.pt"
    torch.save({
        "model_state_dict": best_state,
        "variant": variant, "use_image": use_image, "use_text": use_text,
        "n_continuous": n_cont, "n_genres": n_genres_with_unk,
        "d": D, "nhead": NHEAD, "num_layers": NUM_LAYERS, "dim_ff": DIM_FF,
        "cross_attn_layers": CROSS_ATTN_LAYERS,
        "gate_mode": args.gate_mode, "gate_dropout": GATE_DROPOUT,
        "modality_dropout": args.mod_dropout, "gate_lr_mult": args.gate_lr_mult,
        "scaler": scaler, "genres_train": genres_train,
        "seed": SEED, "best_epoch": best_epoch, "best_val_loss": best_val,
        "val_spearman": float(val_sp),
        "reliability_stats": rel_stats,
        "snapshot_hash": SNAPSHOT_HASH, "split_hashes": split_hashes,
        "git_sha": git_sha, "git_dirty": git_dirty,
        "split_mode": "temporal_no_subs", "drop_subs": True,
    }, ckpt_path)

    test_metrics = {}
    if args.eval_test:
        _, test_preds, test_targets, _ = run_epoch(test_loader, train=False)
        test_sp = spearmanr(test_preds, test_targets)[0]
        has_both_t = len(np.unique((test_targets > 0).astype(int))) == 2
        test_auc = roc_auc_score((test_targets > 0).astype(int), test_preds) if has_both_t else float("nan")
        test_mae = mean_absolute_error(test_targets, test_preds)
        print()
        print(f"=== M6v2 temporal granular TEST (seed={SEED}, variant={variant}) ===")
        print(f"test Spearman: {test_sp:.4f}")
        print(f"test AUC:      {test_auc:.4f}")
        print(f"test MAE:      {test_mae:.4f}")
        test_metrics = {
            "test_spearman": float(test_sp),
            "test_auc": float(test_auc),
            "test_mae": float(test_mae),
        }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps({
            "model": "M6v2_granular_temporal_no_subs",
            "variant": variant,
            "config_tag": tag,
            "gate_mode": args.gate_mode,
            "modality_dropout": args.mod_dropout,
            "gate_lr_mult": args.gate_lr_mult,
            "seed": SEED,
            "split_mode": "temporal_no_subs", "drop_subs": True,
            "cross_attn_layers": CROSS_ATTN_LAYERS,
            "val_spearman": float(val_sp),
            "val_auc": float(val_auc),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "n_params": int(n_params),
            "n_cont": int(n_cont), "n_genres": int(n_genres_with_unk),
            "use_image": use_image, "use_text": use_text,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "reliability_stats": rel_stats,
            "snapshot_hash": SNAPSHOT_HASH,
            "split_hashes": split_hashes,
            "git_sha": git_sha, "git_dirty": git_dirty,
            "test_eval": bool(args.eval_test),
            **test_metrics,
        }) + "\n")
    print(f"\nsaved: {RESULTS}")
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
