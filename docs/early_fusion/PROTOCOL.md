# Early Fusion (RATF) — Research Protocol

Canonical snapshot: `c14dba895034fc4c`  
N = 11,285 rows, 135 channels  
Split: train=9,028, val=1,128, test=1,129 (temporal 80/10/10)

---

## 1. Dataset

### 1.1 Channel Audit

- `channels.json`: 151 channels
- Ingested: **135 channels**
- Missing: 16 channels (verified via API, see `missing_channels_audit.csv`)

**Missing categories:**

- `channel_not_found` (6): @AlexG, @Empleman, @MaxMiller, @NikoOmilana, @ProHomeCooks, @TheEngineeringMindset
- `no_videos_in_window` (4): @CompanyMan, @Hoog, @HowToMakeEverything, @LessonsFromTheScreenplay
- `all_shorts` (4): @Garage54, @PracticalEngineering, @TomSka, @jasontheween
- Unresolved errors (2): @AdamRagusea (timeout), @Kraut (playlist 404)

**Conclusion:** no channels failed due to an ingestion bug. The 16 channels simply had no videos that passed the filtering criteria.

### 1.2 Text Token Length (L)

- Snapshot: 11,285 titles
- Distribution: p50=11, p90=19, p95=22, p99=29, max=174
- **L selected: 32** (0.59% of titles truncated, ≤1% threshold, saves ~200 MB vs L=48)

### 1.3 Thumbnail Size Distribution

| Size            | Count | Percentage | Quality |
| --------------- | ----: | ---------: | ------- |
| 1280×720 (16:9) | 9,565 |      84.8% | maxres  |
| 640×480 (4:3)   | 1,710 |      15.2% | high    |
| 480×360 (4:3)   |    10 |       0.1% | medium  |

- Transform: `squash` → (224, 224), ignoring aspect ratio
- Mode: RGB for all
- Note: 15.3% of thumbnails are non-16:9 → different visual distortion when squashed

### 1.4 Split Hashes

- `train_ids_hash`: `adb6377518e6233e`
- `val_ids_hash`: `1099f7a03511c7ea`
- `test_ids_hash`: `dccdabc759d22895`

---

## 2. Main Baseline (Test Spearman)

**Comparison baseline for RATF.**

| Model                          | Val                 | **Test**            | Role                         |
| ------------------------------ | ------------------- | ------------------- | ---------------------------- |
| M3 (late fusion, LR 2e-5)      | 0.3337              | 0.2957 ± 0.0134     | Old baseline (under-trained) |
| **M3′ (late fusion, LR 1e-3)** | **0.3930 ± 0.0130** | **0.3533 ± 0.0152** | **MAIN BASELINE for RATF**   |

**Context:** M3 → M3′ increased by +0.058 on test (well above the ±0.013 noise). This confirms that the old baseline was under-trained (LR 2e-5 was too small; M0b stopped at epoch 200).

---

## 3. Diagnostic Only (Not a Comparison Baseline)

These models are **not** targets for RATF, but are reported for context and diagnostics.

### 3.1 Tabular Baselines (Test)

| Model                             | Val        | Test       | Notes                         |
| --------------------------------- | ---------- | ---------- | ----------------------------- |
| Mean-reversion (-log1p(trailing)) | 0.0998     | 0.1231     | Trivial baseline              |
| Ridge (alpha=1.0)                 | 0.2313     | 0.1519     | Overfit (val-test gap -0.079) |
| **GBDT (iter=300, lr=0.05)**      | **0.4414** | **0.4120** | **Tabular ceiling**           |
| GBDT_large (iter=1000, lr=0.03)   | 0.4580     | 0.3991     | Mild overfitting              |

**GBDT is the tabular ceiling**, not the target. It is reported for discussion: "multimodal does not outperform pure tabular."

### 3.2 MLP Baselines (Test)

| Model | Tabular Features       | Image | Text | Test Spearman   | AUC    |
| ----- | ---------------------- | ----- | ---- | --------------- | ------ |
| M0    | 4 features             | ❌    | ❌   | 0.2397 ± 0.0170 | 0.6055 |
| M0b   | 12 features (+8 title) | ❌    | ❌   | 0.2323 ± 0.0141 | 0.6114 |

**Decomposition of title-feature effect:** M0b − M0 = −0.0074 (within noise, not helpful).

### 3.3 M4 Ablation Ladder (Val, seed=42)

| Variant             | Tokens | Val Spearman | Val AUC | Best Epoch |
| ------------------- | -----: | -----------: | ------: | ---------: |
| tabular_only        |     13 |   **0.4003** |  0.6694 |         15 |
| no_image (tab+text) |     45 |       0.4036 |  0.6830 |         21 |
| no_text (tab+image) |     63 |       0.3931 |  0.6736 |         14 |
| **full**            |     95 |   **0.3114** |  0.6368 |      **7** |

**Findings:**

- 2-modality ablations (tabular+text / tabular+image) are healthy: ~0.39–0.40.
- **Full 3-modality performance drops by -0.09** and stops at epoch 7. **Anomaly in joint training.**
- **Hypothesis:** attention dilution from 50 image tokens (noisy CLIP patches) + loss of tabular dominance when image+text are added.

---

## 4. RATF Target

**M4/M5/M6 must outperform M3′ test 0.3533.**

- M4 full currently has val = 0.3114 (below M3′ val 0.3930). **Needs improvement.**
- Potential improvements: M4a (pooled tokens), M5 (cross-attention), M6 (reliability gate).
- GBDT 0.4120 is reported as the tabular ceiling, **not** the target.

---

## 5. Research Narrative

> "RATF (token-level early fusion) vs M3′ (tuned late fusion) for relative YouTube performance prediction. GBDT 0.4120 is reported as the tabular ceiling, not the comparison target. Question: can token-level interactions between thumbnails, titles, and metadata outperform late fusion?"

---

## 6. Reproducibility

- Snapshot hash: `c14dba895034fc4c`
- Git SHA M3′: `b7bd737c` (dirty=True at run)
- All runs on GPU (CUDA)
- Seed: 42, 43, 44 (M3′), 42 (M4 ladder, preliminary)
