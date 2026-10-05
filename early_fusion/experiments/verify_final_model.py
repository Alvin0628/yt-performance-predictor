"""Verifikasi model akhir (.pt) sebelum dipakai/dirilis.

Lokasi yang disarankan: early_fusion/experiments/verify_final_model.py

Tiga pemeriksaan:
  1. MUAT   : file .pt dimuat bersih lewat M6Ensemble.load (hanya bergantung pada kode inferensi).
  2. PREDIKSI SAMA : prediksi tiap anggota pada TEST harus sama dengan prediksi yang disimpan run refit
                     (preds/<tag>_seed<N>.npz) -> bukti pengemasan tidak mengubah apa pun.
  3. TABULAR : jalur baris mentah -> tensor (prepare_tabular) harus identik dengan tensor yang dipakai training.

Catatan: pemeriksaan 2 MENGULANG evaluasi test yang sudah tercatat (peran m6_refit); tidak ada informasi
baru dari test, jadi tidak membuka test untuk pemilihan apa pun.

Contoh:
  python early_fusion/experiments/verify_final_model.py \
      --model early_fusion/models/final/m6_granular_ensemble_v1.pt --tag refit_m6_c24_gateoff
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from early_fusion.experiments.m6_core import load_data, iterate_batches, metrics
from early_fusion.experiments._common import load_snapshot
from early_fusion.models.m6_ensemble import M6Ensemble

PRED_TOL = 1e-3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="tag run refit (untuk mencari preds/<tag>_seed<N>.npz)")
    ap.add_argument("--preds-dir", default="early_fusion/results/preds")
    args = ap.parse_args(argv)

    data = load_data(verbose=False)
    dev = data["device"]
    ens = M6Ensemble.load(args.model, device=dev)
    seeds = ens.meta["seeds"]
    print(f"[1/3] MUAT      : OK  ({ens.n_members} anggota, seeds={seeds}, trained_on={ens.meta['fit']})")

    # ---- 2. prediksi test sama dengan yang tersimpan di run refit ----
    store, idx = data["store"], data["test_idx"]
    parts, ys = [], []
    for b in iterate_batches(store, idx, 256, dev, shuffle=False):
        parts.append(ens.predict_batch(b["image_tokens"], b["text_tokens"], b["text_mask"],
                                       b["tabular"], b["genre_idx"]))
        ys.append(b["target"].cpu().numpy())
    members = np.concatenate(parts, axis=1)            # (K, N)
    y = np.concatenate(ys)
    ens_pred = members.mean(axis=0)
    m = metrics(ens_pred, y)
    print(f"      test (ensemble {ens.n_members}): Spearman {m['spearman']:.4f}   AUC {m['auc']:.4f}   MAE {m['mae']:.4f}")

    ok2, max_diffs, saved = True, [], []
    for k, s in enumerate(seeds):
        f = Path(args.preds_dir) / f"{args.tag}_seed{s}.npz"
        if not f.exists():
            print(f"      (lewati) {f} tidak ada; jalankan refit dengan --save-preds")
            ok2 = None
            break
        z = np.load(f)
        d = float(np.max(np.abs(z["test_preds"] - members[k])))
        max_diffs.append(d)
        saved.append(z["test_preds"])
    if ok2 is not None:
        ref_sp = metrics(np.mean(np.stack(saved, axis=0), axis=0), y)["spearman"]
        ok2 = max(max_diffs) < PRED_TOL and abs(ref_sp - m["spearman"]) < 1e-4
        print(f"[2/3] PREDIKSI  : {'OK' if ok2 else 'GAGAL'}  (selisih maks per anggota {max(max_diffs):.2e}, "
              f"Spearman tersimpan {ref_sp:.4f} vs dihitung ulang {m['spearman']:.4f})")

    # ---- 3. jalur baris mentah -> tensor identik dengan training ----
    df, _, _ = load_snapshot(verbose=False)
    sub = df.iloc[idx]
    cont, gidx = ens.prepare_tabular(sub)
    tidx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=store["tab"].device)
    ref_cont = store["tab"][tidx].cpu().numpy()
    ref_gi = store["gi"][tidx].cpu().numpy()
    ok3 = bool(np.allclose(cont, ref_cont, atol=1e-5) and np.array_equal(gidx, ref_gi))
    print(f"[3/3] TABULAR   : {'OK' if ok3 else 'GAGAL'}  (prepare_tabular vs tensor training; "
          f"maks |selisih| {float(np.max(np.abs(cont - ref_cont))):.2e})")

    status = (ok2 is not False) and ok3
    print("\nHASIL VERIFIKASI:", "LULUS" if status else "GAGAL - jangan rilis, kirim output ini ke pengembang")
    return 0 if status else 1


if __name__ == "__main__":
    sys.exit(main())
