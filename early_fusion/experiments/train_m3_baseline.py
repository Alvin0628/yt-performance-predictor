"""
Trains the late fusion head on cached embeddings + tabular features.

Usage:
    python -m models.train

NOTE: not executed end-to-end in the environment that generated this file
(no disk space to install torch here). The model architecture's tensor
shapes were checked by hand and via the standalone shape test in
late_fusion_model.py -- run that first if anything errors here.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from collections import deque
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import json
import random

from features.target import compute_target, invert_target
from models.dataset import VideoDataset, build_tabular_matrix, TABULAR_LOG_COLS, TABULAR_NUMERIC_COLS, TABULAR_BOOL_COLS
from models.late_fusion_model import LateFusionModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, roc_auc_score

from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
    f"{os.environ['POSTGRES_PASSWORD']}@{POSTGRES_HOST}:{POSTGRES_PORT}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

IMAGE_ENCODER = "clip_b32"   # fixed -- no more dinov2/clip_b16 switch
TEXT_ENCODER = "clip"        # fixed -- no more minilm switch
IMAGE_MODE = "squash"        # matches precompute_embeddings.py's hardcoded transform
CLIP_TEXT_MODEL = "openai/clip-vit-base-patch32"
USE_SIM = os.environ.get("USE_SIM", "0") == "1"

_suffix = "_clip_sim" if USE_SIM else "_clip"
SEED = int(os.environ.get("SEED", 42))
CHECKPOINT_PATH = f"early_fusion/models/checkpoints/m3_late_fusion_v1_{IMAGE_ENCODER}{_suffix}.pt"
RESULTS_PATH = "early_fusion/results/m3_results.jsonl"
PLOTS_DIR = "early_fusion/models/plots"
BUNDLE_DIR = "early_fusion/models/bundles"

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 2e-5
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
EMBEDDING_NOISE_STD = 0.02
SPEARMAN_SMOOTHING_WINDOW = 5

def cosine_rows(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return (a * b).sum(axis=1)

def parse_vector(s):
    """Postgres/pgvector returns 'vector' columns as their bracketed text
    form, e.g. '[0.1,0.2,...]', since no pgvector adapter is registered."""
    return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)


def load_data():
    query = """
        SELECT video_id, title, published_at, views, subscriber_count_at_upload,
               genre, duration_seconds, title_length_chars, title_word_count,
               title_capitalized_word_count, title_capitalized_letter_count,
               title_capitalized_letter_ratio, title_symbol_count,
               title_has_question_mark, title_has_number, trailing_avg_views,
               is_first_video, image_embedding, text_embedding
        FROM videos
        WHERE label_finalized = true
          AND image_embedding IS NOT NULL
          AND trailing_avg_views IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df)} rows from Postgres after filtering "
          f"(finalized + valid image + has trailing_avg_views)")

    image_embeddings = np.stack(df["image_embedding"].apply(parse_vector).values)
    text_embeddings = np.stack(df["text_embedding"].apply(parse_vector).values)
    df = df.drop(columns=["image_embedding", "text_embedding"]).reset_index(drop=True)

    if USE_SIM:
        df["clip_sim"] = cosine_rows(image_embeddings, text_embeddings)
        if "clip_sim" not in TABULAR_NUMERIC_COLS:
            TABULAR_NUMERIC_COLS.append("clip_sim")

    df["target"] = compute_target(df["views"], df["trailing_avg_views"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/full_dataset.csv")  # inspection artifact, unchanged

    return df, image_embeddings, text_embeddings

def time_based_split(df):
    sorted_idx = df.sort_values("published_at").index
    n = len(sorted_idx)
    train_end = int(n * (1 - VAL_FRACTION - TEST_FRACTION))
    val_end = int(n * (1 - TEST_FRACTION))
    return (
        sorted_idx[:train_end].to_numpy(),
        sorted_idx[train_end:val_end].to_numpy(),
        sorted_idx[val_end:].to_numpy(),
    )

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(
            batch["image_embedding"].to(device),
            batch["text_embedding"].to(device),
            batch["tabular"].to(device),
        )
        loss = loss_fn(pred, batch["target"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch["target"])
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        pred = model(
            batch["image_embedding"].to(device),
            batch["text_embedding"].to(device),
            batch["tabular"].to(device),
        )
        loss = loss_fn(pred, batch["target"].to(device))
        total_loss += loss.item() * len(batch["target"])
        all_preds.extend(pred.cpu().numpy().tolist())
        all_targets.extend(batch["target"].numpy().tolist())
    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_targets)


def constant_mean_reference(train_targets, val_targets, loss_fn):
    """Huber loss of the 'predict the training mean for everything' baseline.
    If the trained model barely beats this on val loss, it's not learning
    much of a real relationship -- that's underfitting, not a data/architecture
    problem, and no amount of extra capacity or more data fixes it; the fix
    is loosening the optimization (higher LR, less regularization)."""
    const_pred = np.full_like(val_targets, fill_value=train_targets.mean(), dtype=np.float64)
    loss = loss_fn(torch.tensor(const_pred), torch.tensor(val_targets)).item()
    return loss


def plot_training_curves(train_losses, val_losses, val_spearmans, smoothed_spearmans,
                          best_epoch, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)

    # --- Loss curve (overfitting indicator + actual stopping criterion) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, train_losses, label="train_loss")
    ax.plot(epochs_range, val_losses, label="val_loss")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                   label=f"checkpoint (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    fig.tight_layout()
    loss_path = os.path.join(plots_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    # --- Spearman curve (reported for interpretation, not used for stopping) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, val_spearmans, label="val_spearman (raw)", alpha=0.4)
    smoothed_x = [e for e, s in zip(epochs_range, smoothed_spearmans) if s is not None]
    smoothed_y = [s for s in smoothed_spearmans if s is not None]
    if smoothed_y:
        ax.plot(smoothed_x, smoothed_y, label="val_spearman (smoothed)", linewidth=2)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                   label=f"checkpoint (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Validation Spearman Correlation (diagnostic only -- not the stopping criterion)")
    ax.legend()
    fig.tight_layout()
    spearman_path = os.path.join(plots_dir, "spearman_curve.png")
    fig.savefig(spearman_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved training curves to {loss_path} and {spearman_path}")

def export_bundle(*, checkpoint, df, train_idx, test_metrics):
    """Package everything serving needs into one versioned, self-describing
    file: weights, fitted scalers, genre categories, the EXACT tabular column
    order/composition used for this run (TABULAR_NUMERIC_COLS may have had
    'clip_sim' appended by load_data() before we get here -- snapshot it now,
    not the static module list), which encoders produced the cached
    embeddings and how the image was preprocessed (from precompute_embeddings'
    meta.json, since serving has to reproduce that transform exactly), the
    last training-row timestamp (train_end, for the retrain/promotion gate),
    and this run's metrics. Also copies to bundles/latest_<name>.pt for local
    convenience; the model registry (Postgres) takes over "current champion"
    once that exists.

    Not run in this environment (no torch here) -- reviewed by hand against
    build_tabular_matrix()'s concatenation order in models/dataset.py.
    """

    train_end = df.iloc[train_idx]["published_at"].max()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    bundle = {
        "bundle_format_version": 1,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_end": str(train_end),

        "model_state_dict": checkpoint["model_state_dict"],
        "image_dim": checkpoint["image_dim"],
        "text_dim": checkpoint["text_dim"],
        "tabular_dim": checkpoint["tabular_dim"],

        "scaler": checkpoint["scaler"],
        "genre_categories": checkpoint["genre_categories"],
        # Snapshot, not a reference to the shared module list: it may include
        # "clip_sim" appended at runtime by load_data(). This dict is the
        # single source of truth for column order at serving time.
        "feature_columns": {
            "log_cols": list(TABULAR_LOG_COLS),
            "numeric_cols": list(TABULAR_NUMERIC_COLS),
            "bool_cols": list(TABULAR_BOOL_COLS),
        },

        "image_encoder": checkpoint["image_encoder"],
        "text_encoder": checkpoint["text_encoder"],
        "use_sim": checkpoint["use_sim"],
        "image_mode": IMAGE_MODE,
        "clip_text_model": CLIP_TEXT_MODEL,
        
        "metrics": {
            "best_epoch": checkpoint["epoch"],
            "val_loss": checkpoint["val_loss"],
            "val_spearman_raw": checkpoint["val_spearman_raw"],
            "val_spearman_smoothed": checkpoint["val_spearman_smoothed"],
            **test_metrics,
        },
    }

    os.makedirs(BUNDLE_DIR, exist_ok=True)
    versioned_path = os.path.join(BUNDLE_DIR, f"{IMAGE_ENCODER}{_suffix}_{version}.pt")
    torch.save(bundle, versioned_path)

    # NOTE: no longer unconditionally overwrites latest_<name>.pt here --
    # that's now a promotion decision (see promote_if_better()), not an
    # automatic side effect of exporting. Every training run produces a
    # versioned bundle regardless; only a promoted one becomes "latest".
    latest_path = os.path.join(BUNDLE_DIR, f"latest_{IMAGE_ENCODER}{_suffix}.pt")

    # sha256 of the versioned file, so a deploy (or the promotion log) can
    # confirm it copied the bundle it thinks it copied
    with open(versioned_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    print(f"\nSaved model bundle: {versioned_path}")
    print(f"sha256: {digest}")
    return versioned_path, latest_path, bundle


# --- Promotion gate -----------------------------------------------------
#
# export_bundle() always writes a versioned bundle; whether it also becomes
# the one serving/bundle.py actually loads (latest_<name>.pt) is decided
# here, after training, using two gates on this run's own time-based test
# split:
#
#   1. Hard gate -- the new model must beat a trivial linear-on-
#      trailing_avg_views baseline (same idea as models/baseline.py).
#      Catches a training run that's broken outright (bad data, a code
#      regression, a bad batch of embeddings), independent of how it
#      compares to whatever's currently live.
#   2. Soft gate -- a paired bootstrap CI on (new Spearman - old Spearman),
#      both measured on this run's test rows. We only REJECT if the CI's
#      upper bound falls below CI_REJECT_MARGIN, i.e. we're confident the
#      new model is worse by more than that margin. A tie (CI overlapping
#      0, or the small negative band up to the margin) still promotes --
#      trends drift here (audience behavior, channel content, platform
#      dynamics), so a model trained on more recent data is preferred
#      whenever the data can't clearly tell the two apart.
#
# CI_REJECT_MARGIN is the one knob this exposes for now -- adjust directly
# (or via the env var) as you get a feel for how noisy the CI is in
# practice; nothing else here depends on its exact value.
CI_REJECT_MARGIN = float(os.environ.get("PROMOTION_CI_REJECT_MARGIN", "-0.01"))
N_BOOTSTRAP = int(os.environ.get("PROMOTION_N_BOOTSTRAP", "2000"))
PROMOTION_LOG_PATH = "early_fusion/results/promotions_m3.jsonl"


def _linear_baseline_spearman(train_df, test_df):
    """Same trivial baseline as models/baseline.py's run_linear (target ~
    log1p(trailing_avg_views)), inlined here rather than imported to avoid
    a circular import (baseline.py imports from this module). Kept as the
    hard-gate floor: if the fusion model can't beat this, something's
    broken, regardless of the promotion comparison against the live model.
    """
    from sklearn.linear_model import LinearRegression

    X_train = np.log1p(train_df["trailing_avg_views"].values).reshape(-1, 1)
    X_test = np.log1p(test_df["trailing_avg_views"].values).reshape(-1, 1)
    y_train = train_df["target"].values
    y_test = test_df["target"].values

    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    corr, _ = spearmanr(preds, y_test)
    return corr


def _predict_with_bundle(bundle, df, test_idx, image_embeddings, text_embeddings, device):
    """Run a previously-exported bundle's model on this run's test rows,
    reproducing THAT bundle's own scaler / genre_categories / feature-column
    snapshot -- not this run's module-level TABULAR_*_COLS or USE_SIM --
    so the comparison is a faithful like-for-like against what's actually
    deployed, even if features or USE_SIM have changed since it was trained.

    Mirrors build_tabular_matrix() in models/dataset.py, but that function
    reads columns off module globals rather than taking them as an argument,
    so it can't be reused directly for an older bundle's column set.
    """
    test_df = df.iloc[test_idx].copy()
    fc = bundle["feature_columns"]

    if bundle["use_sim"] and "clip_sim" not in test_df.columns:
        test_df["clip_sim"] = cosine_rows(image_embeddings[test_idx], text_embeddings[test_idx])

    log_scaler, numeric_scaler = bundle["scaler"]
    log_cols = test_df[fc["log_cols"]].astype(float).apply(np.log1p).values
    numeric_cols = test_df[fc["numeric_cols"]].astype(float).values
    log_scaled = log_scaler.transform(log_cols)
    numeric_scaled = numeric_scaler.transform(numeric_cols)
    numeric_all_scaled = np.concatenate([log_scaled, numeric_scaled], axis=1)

    bool_cols = test_df[fc["bool_cols"]].astype(float).values

    genre_categories = bundle["genre_categories"]
    genre_onehot = np.zeros((len(test_df), len(genre_categories)), dtype=np.float32)
    for i, genre in enumerate(test_df["genre"].values):
        if genre in genre_categories:
            genre_onehot[i, genre_categories.index(genre)] = 1.0

    tabular = np.concatenate(
        [numeric_all_scaled, bool_cols, genre_onehot], axis=1
    ).astype(np.float32)

    model = LateFusionModel(
        image_dim=bundle["image_dim"], text_dim=bundle["text_dim"], tabular_dim=bundle["tabular_dim"],
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    with torch.no_grad():
        preds = model(
            torch.tensor(image_embeddings[test_idx], dtype=torch.float32, device=device),
            torch.tensor(text_embeddings[test_idx], dtype=torch.float32, device=device),
            torch.tensor(tabular, dtype=torch.float32, device=device),
        ).cpu().numpy()
    return preds


def paired_bootstrap_ci(new_preds, old_preds, targets, n_bootstrap, seed, ci=0.95):
    """CI for (new Spearman - old Spearman) on the shared test set. Each
    resample draws test ROWS (with replacement), not predictions
    independently, so the pairing between the two models is preserved --
    every resample compares both models on the exact same (resampled) rows.
    """
    rng = np.random.default_rng(seed)
    n = len(targets)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        new_corr, _ = spearmanr(new_preds[idx], targets[idx])
        old_corr, _ = spearmanr(old_preds[idx], targets[idx])
        diffs[i] = new_corr - old_corr
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(diffs, [alpha, 1 - alpha])
    return float(lower), float(upper)


def _log_promotion(decision):
    os.makedirs(os.path.dirname(PROMOTION_LOG_PATH), exist_ok=True)
    with open(PROMOTION_LOG_PATH, "a") as f:
        f.write(json.dumps(decision) + "\n")


def _write_airflow_xcom(payload):
    """Hand the promotion decision back to Airflow as this task's XCom.

    Second correction on this function, for the record: retrieve_output=True
    + retrieve_output_path (Docker's get_archive + unpickle) looked like the
    "proper" structured-XCom mechanism, but DockerOperator's
    _attempt_to_retrieve_result() wraps that call in a bare
    `except APIError: return None` with no logging at all -- so when it
    failed (observed: run_training produced no XCom whatsoever, not even
    None), there was no way to see why.

    Falling back to the simpler, already-proven path instead: do_xcom_push
    (xcom_all=False, the default) just returns the last non-empty line of
    container stdout as the XCom value -- exactly what worked in the first
    two real runs, where our human-readable promotion print was correctly
    captured. So: print the decision as one JSON line, and this MUST be the
    absolute last thing main() prints -- nothing may print after this call
    returns, or that later line becomes the XCom instead.
    """
    print(json.dumps(payload))


def promote_if_better(*, versioned_path, latest_path, new_preds, targets,
                       linear_baseline_spearman, df, test_idx,
                       image_embeddings, text_embeddings, device):
    """Decide whether versioned_path should become latest_path (the bundle
    serving/bundle.py actually loads). See the module comment above for the
    two-gate design. Always logs the decision to PROMOTION_LOG_PATH, whether
    promoted or not, for later review.

    Returns the full decision dict (not just a bool) -- the train_model DAG
    hands this whole thing to Airflow as an XCom (see _write_airflow_xcom
    below and dags/train_model.py's check_promotion task), and any future
    logging/alerting module will want the reason/metrics, not just yes-or-no.
    """
    new_spearman, _ = spearmanr(new_preds, targets)
    decision = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "versioned_path": versioned_path,
        "new_spearman": float(new_spearman),
        "linear_baseline_spearman": float(linear_baseline_spearman),
    }

    if not np.isfinite(new_spearman) or new_spearman <= linear_baseline_spearman:
        decision["promoted"] = False
        decision["reason"] = "failed hard gate: did not beat linear (trailing_avg_views) baseline"
        _log_promotion(decision)
        print(f"\n[promotion] REJECTED -- {decision['reason']} "
              f"(new={new_spearman:.4f}, baseline={linear_baseline_spearman:.4f})")
        return decision

    if not os.path.exists(latest_path):
        decision["promoted"] = True
        decision["reason"] = "no currently-served bundle to compare against (first promotion)"
        _log_promotion(decision)
        import shutil
        shutil.copyfile(versioned_path, latest_path)
        print(f"\n[promotion] PROMOTED -- {decision['reason']}")
        return decision

    live_bundle = torch.load(latest_path, map_location=device, weights_only=False)
    old_preds = _predict_with_bundle(live_bundle, df, test_idx, image_embeddings, text_embeddings, device)
    old_spearman, _ = spearmanr(old_preds, targets)

    lower, upper = paired_bootstrap_ci(new_preds, old_preds, targets, N_BOOTSTRAP, seed=SEED)
    decision.update(
        old_spearman=float(old_spearman), ci_lower=lower, ci_upper=upper,
        ci_reject_margin=CI_REJECT_MARGIN, n_bootstrap=N_BOOTSTRAP,
    )

    if upper < CI_REJECT_MARGIN:
        decision["promoted"] = False
        decision["reason"] = "CI confidently below reject margin -- new model is worse than live model"
        _log_promotion(decision)
        print(f"\n[promotion] REJECTED -- {decision['reason']} "
              f"(new={new_spearman:.4f}, old={old_spearman:.4f}, "
              f"CI=[{lower:.4f}, {upper:.4f}], margin={CI_REJECT_MARGIN})")
        return decision

    decision["promoted"] = True
    decision["reason"] = "beat hard gate; not confidently worse than live model (ties go to newer)"
    _log_promotion(decision)
    import shutil
    shutil.copyfile(versioned_path, latest_path)
    print(f"\n[promotion] PROMOTED -- {decision['reason']} "
          f"(new={new_spearman:.4f}, old={old_spearman:.4f}, CI=[{lower:.4f}, {upper:.4f}])")
    return decision


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, image_embeddings, text_embeddings = load_data()

    train_idx, val_idx, test_idx = time_based_split(df)
    print(f"Split (time-based): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())

    train_tabular, scaler = build_tabular_matrix(df.iloc[train_idx], genre_categories, fit_scaler=True)
    val_tabular, _ = build_tabular_matrix(df.iloc[val_idx], genre_categories, scaler=scaler)
    test_tabular, _ = build_tabular_matrix(df.iloc[test_idx], genre_categories, scaler=scaler)

    def make_dataset(idx, tabular):
        sub = df.iloc[idx]
        return VideoDataset(
            image_embeddings[idx], text_embeddings[idx], tabular,
            sub["target"].values, sub["video_id"].values,
        )

    train_ds = make_dataset(train_idx, train_tabular)
    val_ds = make_dataset(val_idx, val_tabular)
    test_ds = make_dataset(test_idx, test_tabular)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = LateFusionModel(
        image_dim=image_embeddings.shape[1],
        text_dim=text_embeddings.shape[1],
        tabular_dim=train_tabular.shape[1],
        dropout=DROPOUT,
        embedding_noise_std=EMBEDDING_NOISE_STD,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.HuberLoss()

    const_val_loss = constant_mean_reference(
        df.iloc[train_idx]["target"].values, df.iloc[val_idx]["target"].values, loss_fn
    )
    print(f"Reference: constant (train-mean) val_loss={const_val_loss:.4f}\n")

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    spearman_window = deque(maxlen=SPEARMAN_SMOOTHING_WINDOW)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    val_spearman_history = []
    smoothed_spearman_history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, loss_fn, device)
        val_spearman, _ = spearmanr(val_preds, val_targets)
        spearman_window.append(val_spearman)
        smoothed_spearman = (
            sum(spearman_window) / len(spearman_window)
            if len(spearman_window) == SPEARMAN_SMOOTHING_WINDOW
            else None
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_spearman_history.append(val_spearman)
        smoothed_spearman_history.append(smoothed_spearman)

        smoothed_str = f"{smoothed_spearman:.4f}" if smoothed_spearman is not None else "N/A"
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_spearman={val_spearman:.4f} | smoothed={smoothed_str}")

        # Pure val_loss-based checkpointing and early stopping: same quantity
        # the optimizer is minimizing, stable/low-variance in our curves,
        # and on the normalized log-ratio scale rather than raw view-count
        # space -- so a single viral outlier can't dominate the decision the
        # way it would if RMSE-in-views were used instead.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "genre_categories": genre_categories,
                "scaler": scaler,
                "image_dim": image_embeddings.shape[1],
                "text_dim": text_embeddings.shape[1],
                "tabular_dim": train_tabular.shape[1],
                "epoch": epoch,
                "val_loss": val_loss,
                "val_spearman_raw": val_spearman,
                "val_spearman_smoothed": smoothed_spearman,
                "image_encoder": IMAGE_ENCODER,
                "text_encoder": TEXT_ENCODER,
                "use_sim": USE_SIM,
            }, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"No val_loss improvement in {EARLY_STOP_PATIENCE} epochs -- stopping early.")
                break

    plot_training_curves(
        train_loss_history, val_loss_history,
        val_spearman_history, smoothed_spearman_history,
        best_epoch, PLOTS_DIR,
    )

    # Final test evaluation using the best checkpoint, not just the last epoch
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, preds, targets = evaluate(model, test_loader, loss_fn, device)

    spearman_corr, _ = spearmanr(preds, targets)

    test_sub = df.iloc[test_idx]
    predicted_views = invert_target(preds, test_sub["trailing_avg_views"].values)
    actual_views = test_sub["views"].values
    rmse_views = np.sqrt(np.mean((predicted_views - actual_views) ** 2))
    mae_target_scale = mean_absolute_error(targets, preds)
    mae_views = mean_absolute_error(actual_views, predicted_views)
    mape_views = mean_absolute_percentage_error(actual_views, predicted_views)
    binary_labels = (targets > 0).astype(int)
    if len(np.unique(binary_labels)) < 2:
        auc = float("nan")
        print("  [warn] test set has only one class (all over- or all under-performing) -- AUC undefined")
    else:
        auc = roc_auc_score(binary_labels, preds)

    print(f"Checkpoint was saved at epoch {checkpoint.get('epoch', '?')} "
          f"(val_loss={checkpoint.get('val_loss', float('nan')):.4f}, "
          f"val_spearman_raw={checkpoint.get('val_spearman_raw', float('nan')):.4f})")

    print(f"\n--- Test results (all downstream/diagnostic -- not used to select the checkpoint) ---")
    print(f"Test loss (Huber, on target scale): {test_loss:.4f}")
    print(f"Spearman correlation (predicted vs actual relative performance): {spearman_corr:.4f}")
    print(f"RMSE in original view-count space: {rmse_views:,.0f}")
    print(f"MAE (target scale): {mae_target_scale:.4f}")
    print(f"MAE (view-count scale): {mae_views:,.0f}")
    print(f"MAPE (view-count scale): {mape_views:.2%}")
    print(f"AUC (overperform vs underperform baseline): {auc:.4f}")
    print(f"(Reference: constant train-mean predictor val_loss was {const_val_loss:.4f} -- "
          f"if best val_loss during training was close to that, revisit LR/regularization.)")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps({
            "image_encoder": IMAGE_ENCODER, "seed": SEED,
            "best_epoch": checkpoint.get("epoch"),
            "val_loss": float(checkpoint.get("val_loss", float("nan"))),
            "test_loss": float(test_loss),
            "test_spearman": float(spearman_corr),
            "test_auc": float(auc),
            "test_mae_target": float(mae_target_scale),
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "text_encoder": TEXT_ENCODER, "use_sim": USE_SIM,
        }) + "\n")

    os.makedirs("early_fusion/results/preds", exist_ok=True)
    np.savez(f"early_fusion/results/preds/{IMAGE_ENCODER}_{TEXT_ENCODER}_sim{int(USE_SIM)}_s{SEED}.npz",
            video_id=test_sub["video_id"].values, pred=preds, target=targets)

    versioned_path, latest_path, _bundle = export_bundle(
        checkpoint=checkpoint,
        df=df,
        train_idx=train_idx,
        test_metrics={
            "test_loss": float(test_loss),
            "test_spearman": float(spearman_corr),
            "test_auc": float(auc),
            "test_mae_target": float(mae_target_scale),
            "test_mae_views": float(mae_views),
            "test_rmse_views": float(rmse_views),
            "test_mape_views": float(mape_views),
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        },
    )

    linear_baseline_spearman = _linear_baseline_spearman(df.iloc[train_idx], test_sub)
    promotion_decision = promote_if_better(
        versioned_path=versioned_path,
        latest_path=latest_path,
        new_preds=preds,
        targets=targets,
        linear_baseline_spearman=linear_baseline_spearman,
        df=df,
        test_idx=test_idx,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        device=device,
    )
    _write_airflow_xcom(promotion_decision)


if __name__ == "__main__":
    main()