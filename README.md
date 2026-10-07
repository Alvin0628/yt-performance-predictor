# yt-performance-predictor

Predicts how a YouTube video will perform **relative to its own channel's recent average**, from the **thumbnail and title alone, before publishing**. It is a deployed service. The model currently served is a token-level **early-fusion** ensemble (RATF); the earlier **late-fusion** model is kept as a fallback.

**Live demo:** https://yt-performance-predictor.vercel.app/

The demo is rate-limited per IP (Vercel Firewall) because each prediction runs CLIP and a 5-model ensemble on a small CPU VM. If you see "Too many predictions", wait a minute.

## Status

| Area | State |
|---|---|
| Ingestion, embeddings, Postgres (pgvector) + MinIO | Done, about 11k labelled rows |
| Early-fusion model (v2, RATF 5-seed ensemble) | **Serving.** Test Spearman 0.431 (temporal split, no subscriber features, see [Results](#results)). Research protocol: [`docs/early_fusion/README.md`](docs/early_fusion/README.md) |
| Late-fusion model (v1) | Kept as a fallback and baseline. Still loadable by the API |
| Train/serve parity checks | Late fusion: `tests/test_bundle_parity.py`. Early fusion: `tests/test_serving_parity.py` and `early_fusion/experiments/verify_final_model.py` (both need the local snapshot, run by hand) |
| Serving (FastAPI, Docker, arm64) | Done, running on an Oracle Cloud Ampere A1 VM |
| HTTPS, request gating (Caddy) | Done: API key, 5 MB body cap |
| Web UI (Next.js on Vercel, separate repo) | Done, with per-IP rate limiting |
| Airflow: ingest, embed, drift monitoring | Done, running on the VM (localhost-only) |
| Airflow: `train_model` (late fusion) + promotion gate | Implemented, **paused** since the switch to the early-fusion model |
| Automatic retraining of the early-fusion model | Not built yet (see [Roadmap](#roadmap)); retraining is manual for now |
| Backups | Not built yet |

## Architecture

```
Browser ──► Vercel (Next.js UI + /api/predict proxy, per-IP rate limit)
                         │  adds X-Api-Key
                         ▼
Oracle VM ─► Caddy (HTTPS, key check, 5 MB cap) ─► FastAPI (CLIP token extractor + early-fusion ensemble)
                                                        ▲ loads the bundle at MODEL_BUNDLE_PATH
Airflow (localhost only)                                │ restarted after a promotion
  ingest_new ─► embed_new ─► train_model (paused) ──────┘
      └──────► monitor_drift
  each task runs in a `ytpp-worker` container against Postgres + MinIO
```

The browser only talks to Vercel, and Vercel talks to the VM. That avoids CORS on the API and keeps the VM address and API key server-side.

## What it predicts

```
target = log(1 + views) - log(1 + trailing_avg_views)
```

`trailing_avg_views` is the mean views of the channel's 5 most recent prior uploads (a video never sees itself). It is recomputed over each channel's full stored history on every ingestion run. Predicted views are recovered with `invert_target`.

## API

`GET /health` returns the loaded model version and a training-end marker.

`POST /predict` (multipart form):

| Field | Notes |
|---|---|
| `thumbnail` | image file |
| `title` | string |
| `trailing_avg_views` | the channel's recent average views |
| `duration_seconds` | planned video length |
| `genre` | YouTube category name |
| `subscriber_count_at_upload` | **optional.** Ignored by the early-fusion model (it does not use subscribers); required only if a late-fusion bundle is loaded |

Response for the early-fusion model: `score` (predicted log-ratio), `expected_views`, `expected_views_low` / `expected_views_high` (score ± one test MAE in log space: a typical-error band, **not** a confidence interval), `member_std` (spread across the 5 ensemble members), `model_version`, `train_end`. `train_end` is a date when `MODEL_TRAIN_END` is set, otherwise the snapshot id. An unreadable image returns 422.

The backend is chosen from the file at `MODEL_BUNDLE_PATH`: a bundle with `model_class == "RATF_M6_Granular_V2"` loads the early-fusion backend, anything else loads the late-fusion one. Rolling back is pointing `MODEL_BUNDLE_PATH` at the old bundle.

## Data

YouTube Data API v3 (`playlistItems.list` + `videos.list`; `search.list` is avoided, it costs about 100x more quota). Hand-picked English-speaking personality, gaming and commentary channels (`pipeline/config/channels.json`), published 2025-01-01 or later, Shorts (180 s or less) excluded. A label counts as final once a video is at least 28 days old at ingestion time.

Stored per video: title, views, duration, publish date, subscriber count, genre (YouTube category), title statistics, trailing average, `is_first_video`, thumbnail (MinIO) and CLIP embeddings (pgvector).

The early-fusion model was trained on a frozen snapshot (`c14dba895034fc4c`): 11,285 videos from 135 channels, split chronologically 80/10/10 (train 9,028, val 1,128, test 1,129). Videos ingested after that snapshot are not used by the served model yet.

## Model v2 (early fusion, RATF): served

RATF is token-level fusion. Instead of one vector per modality, the model sees:

- **Image:** CLIP ViT-B/32 vision tokens (50 x 768: the CLS token plus 49 patches), frozen.
- **Text:** CLIP text-tower token states (32 x 512, plus a padding mask), frozen.
- **Tabular:** one token per feature (trailing average views, duration, title statistics, flags) plus a genre token. **Subscriber count is not used** (removed because of leakage risk; the stored value is the channel's current count, not the count at publish time).
- **Fusion:** two rounds of cross-attention between the three streams, then a joint Transformer (3 layers, d = 192) over a CLS token plus all tokens, then a small MLP head. About 3.0M parameters per member.
- **Reliability gate:** part of the architecture but frozen at identity ("gate-off"), because the learned gate did not help.
- **Ensemble:** 5 members (seeds 100 to 104), each refit on train+val for a fixed 8 epochs. The score is the mean of the 5 predictions.

Serving builds the same inputs as training: `serving/ratf_bundle.py` calls the training token extractor (`early_fusion/datasets/clip_tokens.py`) and the model's own tabular preprocessing, so there is a single source of truth. The bundle is one file holding the config, the scaler, the genre vocabulary and the five state dictionaries. It is **not tracked in git** (about 58 MB); its SHA-256 is in `early_fusion/models/final/m6_granular_ensemble_v1.json`.

### Results

Temporal split without subscriber features, test set of 1,129 videos (`docs/early_fusion/README.md` has the full tables and caveats):

| Model | Seeds | Test Spearman | Test AUC |
|---|---|---|---|
| M3' tuned late fusion (same protocol baseline) | 3 | 0.274 ± 0.017 | 0.620 |
| Early fusion, single model (train only) | 5 | 0.373 ± 0.015 | 0.668 |
| Early fusion, single model (refit train+val) | 5 | 0.379 ± 0.011 | 0.666 |
| **Early fusion, 5-member ensemble (served)** | 5 | **0.431** | 0.690 |

Ensemble MAE on the log-ratio target is 0.476. Caveats worth knowing: the baseline had a smaller tuning budget and about 73x fewer parameters, there is one temporal split, per-seed variance is large, and a Spearman on about 1,100 videos has roughly 0.025 to 0.03 of sampling noise. **Treat the output as a rough ranking signal for comparing options, not a forecast.**

## Model v1 (late fusion): fallback

- **Image / text:** CLIP ViT-B/32 image and text embeddings (512-d each), frozen.
- **Tabular:** log-scaled subscribers and trailing views, duration, title stats, boolean flags, one-hot genre.
- **Head:** each branch is projected to 32-d, then concat, then 64, 16, 1. Only projections and head train.
- **Training:** Huber loss, Adam (lr 2e-5, wd 1e-4), batch 64, dropout 0.2, embedding noise 0.02, early stopping (patience 10). Chronological 80/10/10 split; scalers fit on train only.

Mean over seeds on its own fixed 1,060-row test split (`experiments/results.jsonl`), **with subscriber features**:

| Text encoder | `clip_sim` | Test Spearman | Test AUC |
|---|---|---|---|
| CLIP | off | 0.332 ± 0.016 | 0.643 |
| CLIP | on | 0.333 ± 0.016 | 0.640 |
| MiniLM | off | 0.321 ± 0.012 | 0.641 |
| MiniLM | on | 0.327 ± 0.003 | 0.643 |

Ablation: tabular-only 0.277, then +image 0.302, +text 0.308, full fusion 0.317. Most of the signal comes from channel-level features. **These numbers use a different snapshot, split and feature set than the early-fusion table above, so do not compare them directly.** The like-for-like comparison is the M3' row.

## Retraining and promotion

**Late fusion (`train_model`, paused).** It trains on all embedded, finalized rows and always writes a versioned bundle. Promotion to `latest_clip_b32_clip.pt` needs two gates:

1. **Hard gate:** beat a linear trailing-average baseline.
2. **Soft gate:** a paired bootstrap CI on (new - live) Spearman on the same test rows. Rejected only if the CI's upper bound is below `PROMOTION_CI_REJECT_MARGIN` (default -0.01), so ties go to the newer model.

After a promotion the DAG restarts the `api` container (found by its compose service label) so it loads the new bundle. Decisions are logged to `experiments/promotions.jsonl`. `monitor_drift` (KS tests plus channel-mix shift) is informational only and logs to `experiments/drift_reports.jsonl`.

**Early fusion (manual for now).** The served bundle was trained once on the frozen snapshot; the commands are in `docs/early_fusion/README.md` (section 9.1). Ingestion and embedding keep collecting new data, but it does not reach the served model until automatic retraining exists.

## Security layers

| Layer | Where | Stops |
|---|---|---|
| Per-IP rate limit on `/api/predict` | Vercel Firewall | One visitor flooding the site |
| `X-Api-Key` check on `/predict` | Caddy | Anyone calling the VM directly |
| 5 MB request body cap | Caddy | Oversized uploads |
| Postgres, MinIO, Airflow bound to `127.0.0.1` | Compose | Direct access to backing services |

Known gaps: Airflow uses SimpleAuthManager and mounts `docker.sock` (fine for one person on localhost, reach it only through an SSH tunnel), and worker credentials come from the container environment rather than a secrets backend.

## Configuration

Copy `.env.example` to `.env` and fill it in (never commit `.env`). Notable variables:

| Variable | Purpose |
|---|---|
| `POSTGRES_*`, `MINIO_ROOT_*` | Backing-service credentials |
| `YOUTUBE_API_KEY` | Ingestion |
| `PREDICT_KEY` | Shared secret Caddy requires on `/predict` (same value as in Vercel). Generate with `openssl rand -hex 32` |
| `DOCKER_GID` | `getent group docker \| cut -d: -f3` |
| `AIRFLOW_UID` | `id -u` |
| `HOST_PROJECT_DIR` | Path to this repo as the Docker daemon sees it |
| `AIRFLOW_JWT_SECRET`, `AIRFLOW_FERNET_KEY` | Optional hardening for Airflow |

Variables read by the `api` service (set in `docker-compose.yml` or a `docker-compose.override.yml`):

| Variable | Purpose |
|---|---|
| `MODEL_BUNDLE_PATH` | Bundle to serve, e.g. `/app/models/bundles/m6_granular_ensemble_v1.pt` |
| `MODEL_TRAIN_END` | Optional. Date shown as `train_end` in `/health` and `/predict` (otherwise the snapshot id is shown) |

## Running it

### Production (Oracle VM)

```bash
git pull
docker compose up -d --build
docker compose ps
```

Caddy serves HTTPS for the domain in `Caddyfile` and forwards only requests carrying the correct key. The UI lives in a separate Next.js repo deployed on Vercel with `API_URL` and `PREDICT_KEY` set as environment variables.

Quick checks from any machine:

```bash
curl https://<vm-host>/health                 # 200
curl -i -X POST https://<vm-host>/predict     # 401 without the key
```

### Deploying an early-fusion bundle

The bundle is not in git, so it is copied to the server and checked against its SHA-256.

```bash
# 1. copy it next to the other bundles, then verify
scp m6_granular_ensemble_v1.pt <vm>:~/yt-performance-predictor/models/bundles/
sha256sum models/bundles/m6_granular_ensemble_v1.pt   # compare with early_fusion/models/final/m6_granular_ensemble_v1.json

# 2. point the API at it (docker-compose.override.yml, not committed)
#    services:
#      api:
#        environment:
#          - MODEL_BUNDLE_PATH=/app/models/bundles/m6_granular_ensemble_v1.pt
#          - "MODEL_TRAIN_END=<date of the newest training video>"

# 3. rebuild and restart only the API (about a minute of downtime)
docker compose build api && docker compose up -d --no-deps api

# 4. check
curl https://<vm-host>/health      # model_version should be m6_granular_ensemble_v1
```

To roll back, remove the override (or point `MODEL_BUNDLE_PATH` at the previous bundle) and run `docker compose up -d --no-deps api`. Tag the running image first (`docker tag <api-image>:latest <api-image>:rollback`) if you also want to be able to go back to the old code.

### Local development

```bash
pip install -r requirements-train.txt
cp .env.example .env
docker compose up -d postgres minio
python -m pipeline.run_ingestion
python -m models.precompute_embeddings
python -m models.train
```

To run just the API locally (this compose file publishes port 8000; the production deployment replaces that with `expose` behind Caddy):

```bash
docker compose up -d --build api
curl http://localhost:8000/health
```

### Airflow

```bash
docker compose build worker
docker compose exec postgres psql -U $POSTGRES_USER -l          # "airflow" DB should exist
docker compose run --rm airflow-init
docker compose up -d airflow-api-server airflow-scheduler airflow-dag-processor
docker compose logs airflow-api-server | grep password          # admin password, printed once
```

The UI is on `localhost:8080` (SSH-tunnel it on the VM). Unpause and trigger `ingest_new`, which chains into `embed_new`, `train_model` (late fusion) and `monitor_drift`.

Consistency checks:

```bash
docker compose exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "SELECT count(*), count(image_embedding), count(text_embedding) FROM videos;"
python -m pipeline.check_consistency     # Postgres thumbnail_path vs MinIO objects
```

## CI

`.github/workflows/ci.yml` builds the API image, stages `tests/fixtures/ci_bundle.pt` (a late-fusion bundle) as `latest_clip_b32_clip.pt`, starts `postgres`, `minio` and `api`, and smoke-tests `/health` and `/predict`. Caddy is not started in CI (it needs a real domain). CI therefore exercises the late-fusion path; the early-fusion serving path is checked by `tests/test_serving_parity.py`, which needs the local snapshot, token cache and MinIO, so it is run by hand.

## Known limitations

- The served model is trained on a frozen snapshot and is not retrained automatically yet.
- Its output is a **ranking score**: single members are not well calibrated in absolute level (per-seed MAE 0.48 to 0.56), averaging 5 members brings MAE to 0.476, and the views range returned by the API is a typical-error band, not a guarantee.
- One temporal split and large seed-to-seed variance: single-seed comparisons are unreliable (details and caveats: `docs/early_fusion/README.md`, section 8).
- Late fusion only: `subscriber_count_at_upload` is the channel's **current** count, not the count at publish time (TubeCensus is stubbed). The early-fusion model does not use it.
- `label_finalized` flips at 28 or more days at ingestion time, not exactly day 28.
- `is_first_video` means first video in the fetched 2025+ window, not the channel's first upload.
- The late-fusion promotion test slice moves with each retrain, so that gate is a noisy selection rule, not a fixed holdout.
- The late-fusion backend is CLIP-only, and a late-fusion bundle trained with another encoder is rejected at load time.

## Roadmap

- Automatic retraining of the early-fusion model: incremental token cache, 5-seed refit, a promotion gate for the ensemble, and an API restart after promotion.
- Write the real `train_end` date into the bundle at packaging time (so `MODEL_TRAIN_END` is no longer needed).
- Auto-fill subscribers and average views from a channel handle.
- Nightly off-VM backup of Postgres and MinIO.
- Move credentials to Airflow Connections or a secrets backend.

## Repo structure

```
ingestion/   YouTube client, thumbnail downloader (MinIO), subscriber lookup (stub)
features/    title features, trailing views, target
pipeline/    run_ingestion, check_consistency, monitor_drift, config/
models/      precompute_embeddings, dataset, late_fusion_model, train, baseline, ablations
early_fusion/ RATF model code, token extractor, ensemble inference, experiments, results, split
serving/     FastAPI app, late-fusion bundle loader, early-fusion bundle loader (ratf_bundle)
scripts/     snapshot export, token-cache build/verify, audits
dags/        ingest_new, embed_new, train_model, monitor_drift
db/init/     Postgres schema (pgvector)
docs/        early-fusion research protocol and dataset snapshot notes
tests/       bundle parity tests + CI fixtures
experiments/ results, ablations, promotions, drift reports (jsonl)
```
