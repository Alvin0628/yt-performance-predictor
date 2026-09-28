# yt-performance-predictor

Predicts how a YouTube video will perform **relative to its own channel's recent average**, from the **thumbnail and title alone, before publishing**. It is a continuously retrained, deployed service.

**Live demo:** https://yt-performance-predictor.vercel.app/

The demo is rate-limited per IP (Vercel Firewall) because each prediction runs CLIP on a small CPU VM. If you see "Too many predictions", wait a minute.

## Status

| Area | State |
|---|---|
| Ingestion, embeddings, late-fusion model, training | Done. Postgres (pgvector) + MinIO, about 11k rows, test Spearman about 0.30 to 0.33 |
| Model bundle export + train/serve parity test | Done, checked in CI and by `tests/test_bundle_parity.py` |
| Serving (FastAPI, Docker, arm64) | Done, running on an Oracle Cloud Ampere A1 VM |
| HTTPS, request gating (Caddy) | Done: API key, 5 MB body cap |
| Web UI (Next.js on Vercel, separate repo) | Done, with per-IP rate limiting |
| Airflow: ingest, embed, train, drift monitoring | Done, running on the VM (localhost-only) |
| Promotion gate (hard baseline gate + paired bootstrap) | Done, logged to `experiments/promotions.jsonl` |
| Backups | Not built yet |

## Architecture

```
Browser ──► Vercel (Next.js UI + /api/predict proxy, per-IP rate limit)
                         │  adds X-Api-Key
                         ▼
Oracle VM ─► Caddy (HTTPS, key check, 5 MB cap) ─► FastAPI (CLIP + late-fusion head)
                                                        ▲ loads models/bundles/latest_*.pt
Airflow (localhost only)                                │ restarted after a promotion
  ingest_new ─► embed_new ─► train_model ───────────────┘
      └──────► monitor_drift
  each task runs in a `ytpp-worker` container against Postgres + MinIO
```

The browser only talks to Vercel, and Vercel talks to the VM. That avoids CORS on the API and keeps the VM address and API key server-side.

## What it predicts

```
target = log(1 + views) - log(1 + trailing_avg_views)
```

`trailing_avg_views` is the mean views of the channel's 5 most recent prior uploads (a video never sees itself). It is recomputed over each channel's full stored history on every ingestion run. Predicted views are recovered with `invert_target`.

## Data

YouTube Data API v3 (`playlistItems.list` + `videos.list`; `search.list` is avoided, it costs about 100x more quota). Hand-picked English-speaking personality, gaming and commentary channels (`pipeline/config/channels.json`), published 2025-01-01 or later, Shorts (180 s or less) excluded. A label counts as final once a video is at least 28 days old at ingestion time.

Stored per video: title, views, duration, publish date, subscriber count, genre (YouTube category), title statistics, trailing average, `is_first_video`, thumbnail (MinIO) and CLIP embeddings (pgvector).

## Model (v1, late fusion)

- **Image:** CLIP ViT-B/32 (512-d), frozen.
- **Text:** CLIP text tower (512-d), frozen.
- **Tabular:** log-scaled subscribers and trailing views, duration, title stats, boolean flags, one-hot genre.
- **Head:** each branch is projected to 32-d, then concat, then 64, 16, 1. Only projections and head train.
- **Training:** Huber loss, Adam (lr 2e-5, wd 1e-4), batch 64, dropout 0.2, embedding noise 0.02, early stopping (patience 10). Chronological 80/10/10 split; scalers fit on train only.

### Results

Mean over seeds on a fixed 1,060-row test split (`experiments/results.jsonl`):

| Text encoder | `clip_sim` | Test Spearman | Test AUC |
|---|---|---|---|
| CLIP | off | 0.332 ± 0.016 | 0.643 |
| CLIP | on | 0.333 ± 0.016 | 0.640 |
| MiniLM | off | 0.321 ± 0.012 | 0.641 |
| MiniLM | on | 0.327 ± 0.003 | 0.643 |

Ablation: tabular-only 0.277, then +image 0.302, +text 0.308, full fusion 0.317. Most of the signal comes from channel-level features, and the thumbnail and title add about +0.04. Face, OCR and colour-stat features gave no measurable benefit and were dropped. **Treat the output as a rough ranking signal for comparing options, not a forecast.**

## Retraining and promotion

`train_model` trains on all embedded, finalized rows and always writes a versioned bundle. Promotion to `latest_clip_b32_clip.pt` (the file the API serves) needs two gates:

1. **Hard gate:** beat a linear trailing-average baseline.
2. **Soft gate:** a paired bootstrap CI on (new − live) Spearman on the same test rows. Rejected only if the CI's upper bound is below `PROMOTION_CI_REJECT_MARGIN` (default -0.01), so ties go to the newer model.

After a promotion the DAG restarts the `api` container so it loads the new bundle. Decisions are logged to `experiments/promotions.jsonl`. `monitor_drift` (KS tests plus channel-mix shift) is informational only and logs to `experiments/drift_reports.jsonl`.

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

### Local development

```bash
pip install -r requirements-train.txt
cp .env.example .env
docker compose up -d postgres minio
python -m pipeline.run_ingestion
python -m models.precompute_embeddings
python -m models.train
```

To run just the API locally with a port published (the CI override does this):

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.ci.yml docker compose up -d --build api
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

The UI is on `localhost:8080` (SSH-tunnel it on the VM). Unpause and trigger `ingest_new`, which chains into `embed_new`, `train_model` and `monitor_drift`.

Consistency checks:

```bash
docker compose exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "SELECT count(*), count(image_embedding), count(text_embedding) FROM videos;"
python -m pipeline.check_consistency     # Postgres thumbnail_path vs MinIO objects
```

## CI

`.github/workflows/ci.yml` builds the API image, starts only `api` with the bundle in `tests/fixtures/`, and smoke-tests `/health` and `/predict`. `docker-compose.ci.yml` publishes port 8000 for that purpose only. Caddy is not started in CI (it needs a real domain).

## Known limitations

- `subscriber_count_at_upload` is the channel's **current** count, not the count at publish time (TubeCensus is stubbed).
- `label_finalized` flips at 28 or more days at ingestion time, not exactly day 28.
- `is_first_video` means first video in the fetched 2025+ window, not the channel's first upload.
- The promotion test slice moves with each retrain, so the gate is a noisy selection rule, not a fixed holdout.
- Serving is CLIP-only, and a bundle trained with another encoder is rejected at load time.

## Roadmap

- Auto-fill subscribers and average views from a channel handle.
- Nightly off-VM backup of Postgres and MinIO.
- Move credentials to Airflow Connections or a secrets backend.
- v2: early-fusion cross-attention over image patches and title tokens (needs 30k+ rows and a GPU).

## Repo structure

```
ingestion/   YouTube client, thumbnail downloader (MinIO), subscriber lookup (stub)
features/    title features, trailing views, target
pipeline/    run_ingestion, check_consistency, monitor_drift, config/
models/      precompute_embeddings, dataset, late_fusion_model, train, baseline, ablations
serving/     FastAPI app, bundle loader, feature reconstruction
dags/        ingest_new, embed_new, train_model, monitor_drift
db/init/     Postgres schema (pgvector)
tests/       bundle parity test + CI fixtures
experiments/ results, ablations, promotions, drift reports (jsonl)
```