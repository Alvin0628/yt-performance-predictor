"""Test additional signal from CLIP embeddings (image+text).

Goal: check whether tabular + embedding > tabular only.
On 2 splits: temporal + unseen-channel.

If tabular+emb >> tabular only on unseen-channel → image+text IS USEFUL.
If there is no difference → image+text is not useful, M5/M6 are pointless.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

from features.target import compute_target
from models.dataset import build_tabular_matrix


SNAPSHOT = Path("data_snapshots/snapshot.parquet")
RANDOM_STATE = 42
PCA_DIM = 32


def parse_vector(s):
    if isinstance(s, str):
        return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(s, dtype=np.float32)


def load_embeddings(df):
    img_emb = np.stack(df["image_embedding"].apply(parse_vector).values)
    txt_emb = np.stack(df["text_embedding"].apply(parse_vector).values)
    return img_emb, txt_emb


def eval_config(df, img_emb, txt_emb, train_idx, test_idx, label):
    """Train GBDT with various feature configurations."""
    df = df.reset_index(drop=True)
    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    genres_train = sorted(df.loc[train_idx]["genre"].dropna().unique().tolist())
    tab_tr, sc = build_tabular_matrix(df.loc[train_idx], genres_train, fit_scaler=True)
    tab_te, _ = build_tabular_matrix(df.loc[test_idx], genres_train, scaler=sc)

    y_tr = df["target"].values[train_idx]
    y_te = df["target"].values[test_idx]

    # PCA on train only
    pi = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(img_emb[train_idx])
    pt = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(txt_emb[train_idx])

    img_tr_pca = pi.transform(img_emb[train_idx])
    img_te_pca = pi.transform(img_emb[test_idx])
    txt_tr_pca = pt.transform(txt_emb[train_idx])
    txt_te_pca = pt.transform(txt_emb[test_idx])

    configs = {
        "tabular_only":              (tab_tr, tab_te),
        "tabular + img":             (np.hstack([tab_tr, img_tr_pca]), np.hstack([tab_te, img_te_pca])),
        "tabular + txt":             (np.hstack([tab_tr, txt_tr_pca]), np.hstack([tab_te, txt_te_pca])),
        "tabular + img + txt":       (np.hstack([tab_tr, img_tr_pca, txt_tr_pca]),
                                     np.hstack([tab_te, img_te_pca, txt_te_pca])),
        "img + txt only":            (np.hstack([img_tr_pca, txt_tr_pca]),
                                     np.hstack([img_te_pca, txt_te_pca])),
    }

    print(f"\n--- {label} ---")
    print(f"  n_train={len(train_idx)}, n_test={len(test_idx)}")
    print(f"  train channels unique={df.loc[train_idx]['channel_id'].nunique()}, "
          f"test channels unique={df.loc[test_idx]['channel_id'].nunique()}")

    results = {}
    for name, (X_tr, X_te) in configs.items():
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                           random_state=RANDOM_STATE).fit(X_tr, y_tr)
        sp = spearmanr(m.predict(X_te), y_te)[0]
        results[name] = sp
        print(f"  {name:<25} dim={X_tr.shape[1]:<3} Spearman = {sp:.4f}")

    return results


def main():
    df = pd.read_parquet(SNAPSHOT).reset_index(drop=True)
    df["target"] = compute_target(df["views"], df["trailing_avg_views"])
    img_emb, txt_emb = load_embeddings(df)
    print(f"snapshot: {len(df)} rows, {df['channel_id'].nunique()} channels")
    print(f"img_emb: {img_emb.shape}, txt_emb: {txt_emb.shape}")
    print(f"PCA_DIM: {PCA_DIM}")
    print()

    # ============ Temporal Split ============
    print("=" * 70)
    print("TEMPORAL SPLIT (test channel is present in train)")
    print("=" * 70)
    tr_t = np.where(df["split"].values == "train")[0]
    te_t = np.where(df["split"].values == "test")[0]
    results_temp = eval_config(df, img_emb, txt_emb, tr_t, te_t, "Temporal")

    # ============ Unseen-Channel Split ============
    print()
    print("=" * 70)
    print("UNSEEN-CHANNEL SPLIT (test channel is NOT present in train)")
    print("=" * 70)
    # Use train+val only (test remains untouched)
    train_va_idx = np.where(df["split"].values != "test")[0]
    df_tv = df.loc[train_va_idx].reset_index(drop=True)
    img_tv = img_emb[train_va_idx]
    txt_tv = txt_emb[train_va_idx]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_u, te_u = next(gss.split(df_tv, groups=df_tv["channel_id"].values))

    results_unseen = eval_config(df_tv, img_tv, txt_tv, tr_u, te_u, "Unseen-channel")

    # ============ Summary ============
    print()
    print("=" * 70)
    print("SUMMARY — Delta from tabular_only")
    print("=" * 70)
    print(f"{'Config':<25} {'Temporal':<12} {'Unseen':<12} {'Delta_unseen':<12}")
    for name in ["tabular_only", "tabular + img", "tabular + txt", "tabular + img + txt", "img + txt only"]:
        t = results_temp[name]
        u = results_unseen[name]
        delta = u - results_unseen["tabular_only"]
        print(f"{name:<25} {t:<12.4f} {u:<12.4f} {delta:+.4f}")

    print()
    print("INTERPRETATION:")
    delta_img_txt_unseen = results_unseen["tabular + img + txt"] - results_unseen["tabular_only"]
    if delta_img_txt_unseen > 0.05:
        print(f"  ✅ Embedding ADDS signal on unseen-channel (+{delta_img_txt_unseen:.4f}).")
        print(f"     M5/M6 HAVE STRONG POTENTIAL.")
    elif delta_img_txt_unseen > 0.02:
        print(f"  ⚠️  Embedding adds a small amount on unseen-channel (+{delta_img_txt_unseen:.4f}).")
    else:
        print(f"  ❌ Embedding does not add on unseen-channel ({delta_img_txt_unseen:+.4f}).")
        print(f"     M5/M6 are likely pointless.")

    # Also check img+txt only (without tabular)
    print()
    print(f"  Sanity: img+txt only on unseen = {results_unseen['img + txt only']:.4f}")
    print(f"         (if > 0.20, embedding itself has signal)")


if __name__ == "__main__":
    main()