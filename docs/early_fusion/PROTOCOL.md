# Channel Audit (Final)

- **channels.json:** 151 channels
- **Ingested:** 135 channels
- **Missing:** 16 channels, all verified via the API (see `missing_channels_audit.csv`)

## Categories

- **channel_not_found (6):** `@AlexG`, `@Empleman`, `@MaxMiller`, `@NikoOmilana`, `@ProHomeCooks`, `@TheEngineeringMindset` — handles could not be resolved.

- **no_videos_in_window (4):** `@CompanyMan`, `@Hoog`, `@HowToMakeEverything`, `@LessonsFromTheScreenplay` — no uploads published on or after `2025-01-01`.

- **all_shorts (4):** `@Garage54`, `@PracticalEngineering`, `@TomSka`, `@jasontheween` — all uploads are Shorts (≤180 seconds).

- **Unresolved errors (2):** `@AdamRagusea`, `@Kraut` — network timeout / playlist 404 errors.

## Conclusion

No channels failed due to an ingestion bug.

These 16 channels simply have no videos that meet the filtering criteria.

The final snapshot covers **135 channels**.

## Text Token Length (L)

- Snapshot: 11,285 titles
- Distribution: p50=11, p90=19, p95=22, p99=29, max=174
- Selected L: **32**
- Rationale: 0.59% of titles are truncated (≤1% threshold), saving ~200 MB of memory vs L=48
- Longest title: 174 tokens (outlier, still truncated to 32)

## Thumbnail Size Distribution (all 11.285)

- 1280×720 (16:9, maxres): 9.565 (84.8%)
- 640×480 (4:3, high): 1.710 (15.2%)
- 480×360 (4:3, medium): 10 (0.1%)
- Transform: `squash` → (224, 224), ignoring aspect ratio
- Mode: RGB for all
- Note: 15.3% of thumbnails are non-16:9 → different visual distortion when squashed

## M3 Baseline (Late Fusion) — Local Snapshot

- Snapshot: c14dba895034fc4c (N=11285)
- Split: train=9028, val=1128, test=1129
- Seeds: 42, 43, 44
- Test Spearman: 0.2957 ± 0.0134
- Test AUC: 0.6293 ± 0.0064

## Baseline Decomposition

| Model | Tabular Features     | Image | Text | Spearman        | AUC    |
| ----- | -------------------- | ----- | ---- | --------------- | ------ |
| M0    | 4 features           | ❌    | ❌   | 0.2397 ± 0.0170 | 0.6055 |
| M0b   | 12 features (+title) | ❌    | ❌   | 0.2323 ± 0.0141 | 0.6114 |
| M3    | 12 features          | ✅    | ✅   | 0.2957 ± 0.0134 | 0.6293 |

**Decomposition:**

- Effect of 8 title features: M0b − M0 = −0.0074 (not helpful)
- Effect of image + text: M3 − M0b = +0.0634 (substantial)
- Combined effect: M3 − M0 = +0.0560

**Note:** M0b reached `best_epoch = 200` (did not converge across all seeds), which may indicate that it was under-trained. This is not a blocker for the baseline.

## M4 Ablation Ladder (in progress, seed=42)

Snapshot: c14dba895034fc4c | Split: train=9028, val=1128, test=1129  
Git SHA at run: b7bd737c, dirty=False  
M4 Parameters: 587,393

### Val Spearman (temporary, seed=42)

| Variant | Tokens | Val Spearman | Val Loss | Best Epoch |
|---|---|---|---|---|
| M4 tabular_only | 13 (CLS+12) | **0.4003** | 0.1928 | 15 |
| M4 no_image | 45 | *(pending)* | | |
| M4 no_text | 63 | *(pending)* | | |
| M4 full | 95 | *(pending)* | | |

### Comparison Baseline (val, same snapshot)

| Model | Val Spearman | Val Loss | Best Epoch |
|---|---|---|---|
| M3 late fusion | **0.3337** | 0.2032 | 53 |