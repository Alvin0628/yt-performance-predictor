"""
Loader + single-request inference for the early-fusion RATF M6 ensemble
(file produced by early_fusion/experiments/package_final_model.py).

Same public surface as serving.bundle.LoadedBundle, so serving/app.py can use
either one:
    .bundle["version"], .bundle["train_end"]      (used by /health and startup log)
    .predict(thumbnail=..., title=..., ...)       -> dict

Preprocessing is NOT re-implemented here. It calls the same functions that built
the training token cache (early_fusion/datasets/clip_tokens.py) and the same
tabular path training used (M6Ensemble.prepare_tabular -> models.dataset.
build_tabular_matrix), so there is one source of truth.

Parity details that matter:
  * image: convert("RGB") -> squash 224x224 transform -> CLIP vision last_hidden_state (1, 50, 768)
  * text : CLIP tokenizer, padding="max_length", max_length=32 -> text_model last_hidden_state (1, 32, 512)
  * both token tensors go through an fp16 round-trip, because training read them
    from an fp16 memmap cache
  * subscriber count is dropped by the model; a placeholder is only needed so the
    scaler (fit with that column) can run, its scaled value is deleted afterwards
"""

import io

import numpy as np
import pandas as pd
import torch
from PIL import Image

from features.target import invert_target
from features.title_features import title_features
from early_fusion.datasets.clip_tokens import (
    MAX_TEXT_TOKENS, image_tokens, load_clip, text_tokens,
)
from early_fusion.models.m6_ensemble import M6Ensemble

RATF_MODEL_CLASS = "RATF_M6_Granular_V2"
SUBS_PLACEHOLDER = 0.0


class LoadedRatfBundle:
    uses_subscribers = False

    def __init__(self, bundle_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ens = M6Ensemble.load(bundle_path, device=self.device)
        meta = self.ens.meta
        if meta.get("model_class") != RATF_MODEL_CLASS:
            raise ValueError(f"unexpected model_class {meta.get('model_class')!r}, expected {RATF_MODEL_CLASS!r}")

        self.clip, _encode_fn, self.transform, self.tokenizer, _cfg = load_clip(self.device)

        metrics = meta.get("refit_metrics") or {}
        self.typical_log_error = metrics.get("test_mae_ensemble")  # MAE in log-ratio space, may be None
        # Same shape as the late-fusion bundle dict so app.py's /health and startup log work unchanged.
        # The packaged file has no train_end date; fall back to the snapshot id.
        self.bundle = {
            "version": meta.get("name", "m6_granular_ensemble"),
            "train_end": meta.get("train_end") or f"snapshot:{meta.get('snapshot_hash')}",
            "model_class": meta["model_class"],
            "n_members": self.ens.n_members,
        }

    @staticmethod
    def _load_image(thumbnail):
        if isinstance(thumbnail, Image.Image):
            img = thumbnail
        elif isinstance(thumbnail, (bytes, bytearray)):
            img = Image.open(io.BytesIO(thumbnail))
        else:
            img = Image.open(thumbnail)
        return img.convert("RGB")

    def _tabular_row(self, title, trailing_avg_views, duration_seconds, genre):
        row = {
            "subscriber_count_at_upload": SUBS_PLACEHOLDER,
            "trailing_avg_views": float(trailing_avg_views),
            "duration_seconds": float(duration_seconds),
            **title_features(title),
            "genre": genre or "",
        }
        return pd.DataFrame([row])

    @torch.inference_mode()
    def predict(self, *, thumbnail, title, trailing_avg_views, duration_seconds, genre,
                subscriber_count_at_upload=None):
        # subscriber_count_at_upload is accepted for API compatibility and ignored.
        title = title or ""
        image = self._load_image(thumbnail)

        x = self.transform(image).unsqueeze(0).to(self.device)
        img_tok = image_tokens(self.clip, x)
        txt_tok, txt_mask, _ids = text_tokens(self.clip, self.tokenizer, [title], self.device, MAX_TEXT_TOKENS)

        # fp16 round-trip = what the training cache stored
        img_tok = img_tok.half().float().cpu().numpy()
        txt_tok = txt_tok.half().float().cpu().numpy()
        txt_mask = txt_mask.bool().cpu().numpy()

        cont, gidx = self.ens.prepare_tabular(
            self._tabular_row(title, trailing_avg_views, duration_seconds, genre)
        )
        mean, members = self.ens.predict(img_tok, txt_tok, txt_mask, cont, gidx, return_members=True)

        score = float(mean[0])
        expected = float(invert_target(score, trailing_avg_views))
        out = {
            "score": score,
            "expected_views": expected,
            "member_std": float(members[:, 0].std()),
            "model_version": self.bundle["version"],
            "train_end": self.bundle["train_end"],
        }
        if self.typical_log_error is not None:
            e = float(self.typical_log_error)
            # +/- one test MAE in log-ratio space -- a typical-error band, not a confidence interval
            out["expected_views_low"] = float(invert_target(score - e, trailing_avg_views))
            out["expected_views_high"] = float(invert_target(score + e, trailing_avg_views))
        return out
