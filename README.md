# yt-performance-predictor

Predicts a YouTube video's relative performance (views vs. the channel's own recent baseline) from **thumbnail + title alone, before publish**. Built as a continuously-retrained, deployed service.

## Status

| Area | State |
|---|---|
| Ingestion, embeddings, late-fusion model + training | Built. Now on Postgres + MinIO (see below), ~10.6k rows, test Spearman ≈ 0.33 (see [Results](#results)) |
| Data storage: Postgres (`videos` table, pgvector) + MinIO (thumbnails) | **Built** — ingestion, embedding precompute, and training all read/write Postgres directly; thumbnails live in a MinIO bucket, not on local disk. Replaces the old `videos.csv` + `data/images/` flow. |
| Model bundle export + train/serve parity test | Built and passing as of the last recorded run (0/200 samples outside tolerance) |
| Serving (FastAPI, Dockerfile, requirements-serve.txt) | Built and validated locally (amd64) — containerized CPU predictions match the training path exactly |
| Docker Compose stack (api + Caddy + Postgres + MinIO) | Builds and starts cleanly locally. **See the open item below** — `api`'s port is currently published straight to the host, which wasn't true when this was last checked in. |
| Deployment (Oracle Cloud VM, Caddy, arm64) | Still blocked on Oracle Always Free Ampere A1 capacity; `Caddyfile` still points at the placeholder `your-domain.example.com`; arm64 still not tested on real hardware |
| Airflow, retrain/promotion, drift monitoring | Designed only, not built (see [Roadmap](#roadmap)) |

### ⚠️ Open item to resolve before deploying

`docker-compose.yml`'s `api` service now has `ports: ["8000:8000"]` (and `.github/workflows/ci.yml` curls it at `localhost:8000` directly), instead of the `expose: ["8000"]`-only setup this README previously documented as load-bearing for production ("nobody should be able to skip Caddy and hit the model directly"). This may just be the temporary local-testing change from the old workflow that never got reverted — if so, switch it back to `expose` before this goes anywhere near the internet. If it's intentional now, this section needs a real explanation of why, and the security note below needs rewriting.

Relatedly, `requirements-serve.txt` now pulls in `sqlalchemy`, `psycopg2-binary`, and `boto3`, and the `api` container `depends_on` Postgres and MinIO being healthy — but nothing under `serving/` (`app.py`, `bundle.py`, `features.py`) actually imports or connects to either. Worth trimming, or wiring up if there's a reason serving is meant to reach them.

## What it predicts

```
target = log(1 + views) − log(1 + trailing_avg_views)
```

`trailing_avg_views` = mean views over the channel's 5 most recent prior uploads (shifted so a video never sees itself). Predicted views recovered via `invert_target`. Since the move to Postgres, this is now recomputed over each channel's **complete** stored history on every ingestion run (existing rows + new), not just the rows touched in that run.

## Pipeline

```
YouTube Data API v3 ──► pipeline/run_ingestion.py ──► Postgres `videos` table + MinIO thumbnails bucket
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
   models/precompute_embeddings.py          (visual sub-features: faces, OCR, color
   (frozen CLIP/DINOv2/MiniLM,                — dropped, see Ablations)
    reads/writes embeddings to Postgres)
                 └───────────────────┬───────────────────┘
                                     ▼
                             models/train.py
                (reads from Postgres, late-fusion head, chronological split,
                 logs to experiments/results.jsonl)
```

`pipeline/check_consistency.py` is a read-only reconciliation script: it compares `videos.thumbnail_path` rows in Postgres against what's actually in the MinIO `thumbnails` bucket and reports mismatches (missing objects, orphaned objects). Doesn't fix anything, just reports.

## Data

YouTube Data API v3 (`playlistItems.list` + `videos.list`; `search.list` avoided — ~100x more quota). Hand-curated English-speaking personality/gaming/commentary channels (excludes news/release-driven channels), published 2025-01-01+, shorts (≤180s) excluded.

As of the last full evaluation sweep: 10,591 usable rows (8,472 train / 1,059 val / 1,060 test) — this is what the [Results](#results) table below is measured on. Ingestion has continued since then (currently ~10,625 rows); a single spot-check run on the larger dataset gave test Spearman 0.321 (CLIP, no `clip_sim`), consistent with the range below, but the full multi-seed sweep hasn't been rerun on the new size yet.

**Retrieved:** thumbnail, title, views, duration, publish date, subscriber count, `categoryId` → genre.
**Derived:** title stats (length, caps, symbols, question mark, has-number), trailing average views, `is_first_video`, optional `clip_sim` (CLIP image/title cosine similarity).

## Model: v1 late fusion

- **Image:** CLIP ViT-B/32 (512-d), frozen. Alternatives: DINOv2 ViT-S/14, CLIP ViT-B/16.
- **Text:** CLIP text tower (512-d), frozen. Alternative: `all-MiniLM-L6-v2`.
- **Tabular:** log-scaled subscriber count + trailing views, duration, title stats, boolean flags, one-hot genre, optional `clip_sim`.
- **Fusion:** each branch → 32-d projection → concat → 64 → 16 → 1. Only projections + head train; embeddings precomputed once.
- **Training:** Huber loss, Adam (lr 2e-5, wd 1e-4), batch 64, dropout 0.2, embedding noise (std 0.02), early stopping (patience 10, max 200 epochs). Chronological 80/10/10 split by `published_at`; scalers fit on train only.

### Results

Mean ± std over seeds, same 1,060-row test split (`experiments/results.jsonl`, `n_train=8472` rows):

| Text encoder | `clip_sim` | Test Spearman | Test AUC |
|---|---|---|---|
| CLIP | off | 0.332 ± 0.016 | 0.643 |
| CLIP | on | 0.333 ± 0.016 | 0.640 |
| MiniLM | off | 0.321 ± 0.012 | 0.641 |
| MiniLM | on | 0.327 ± 0.003 | 0.643 |

Differences between encoders/`clip_sim` are within seed noise.

### Ablations

**Modality** (single seed): tabular-only 0.277 Spearman → +image 0.302 → +text 0.308 → full fusion 0.317. Most skill comes from channel-level features; thumbnail+title add ~+0.04.

**Visual tabular features** (5 seeds, embeddings fixed, `experiments/ablation_visual_flags.jsonl`): dropping face/OCR/color-stat columns showed no measurable benefit (Δ within noise, non-monotonic across seeds). **Decision: dropped all 7 from training and serving.**

## Known limitations

- `subscriber_count_at_upload` is still the *current* count, not at-publish — `ingestion/tubecensus_client.py` remains an explicit stub returning the fallback (current) count; TubeCensus integration is still shelved (Windows permission issues, storage cost). Tracked as a deliberate approximation, not a bug.
- Label timing isn't a fixed horizon — `label_finalized` still flips at ≥28 days *at ingestion time*, not exactly day 28.
- `is_first_video` still means first video in the fetched 2025+ window, not the channel's actual first upload — the `PUBLISHED_AFTER` cutoff in `pipeline/run_ingestion.py` is unchanged.
- ~~`compute_trailing_views` may compute a truncated history on incremental re-runs~~ — **fixed** by the Postgres migration: each run now recomputes over the channel's full stored history (existing + new rows), not just what that run touched.

## Serving

`export_bundle()` (in `train.py`) packages weights, scalers, feature-column order, and encoder config into one versioned file; `tests/test_bundle_parity.py` checks the serving path reproduces it (passing as of the last recorded run, max diff 2.8e-3 vs. a 1e-2 bug threshold). Serving is **CLIP-only** — a bundle trained with a different encoder is rejected at load time.

Serving itself (`serving/app.py`, `serving/bundle.py`, `serving/features.py`) is stateless: it loads a bundle file once at startup and serves predictions from memory, with no direct dependency on Postgres or MinIO. (See the open item above about `requirements-serve.txt` and the compose file currently suggesting otherwise.)

### Running it — Docker Compose (primary workflow)

This is how the app actually runs, both locally and on the eventual Oracle VM: FastAPI behind Caddy, with Postgres + MinIO as backing services for the ingestion/training side, all on Caddy's internal Docker network.

```bash
docker compose up -d --build
docker compose ps
```

Copy `.env.example` to `.env` and fill in `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` and `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` first — `docker-compose.yml` reads these for the `postgres` and `minio` services (and passes them through to `api`, even though `api` doesn't currently use them — see the open item above).

`Caddyfile` is currently keyed to a placeholder domain (`your-domain.example.com`), so Caddy itself won't respond to `localhost` until that's swapped for a real domain pointed at the VM. Since there's no domain to test against yet, Caddy itself can't be meaningfully exercised locally — only `api` can be.

**Testing `/predict` locally:** `api`'s port is currently published directly (`ports: ["8000:8000"]`), so `curl http://localhost:8000/...` works against it directly from the host right now, bypassing Caddy — see the open item above about whether that's intentional. If it gets reverted to `expose: ["8000"]`-only, test the API from inside the Compose network instead:

```bash
docker compose exec caddy wget -qO- http://api:8000/health
```

### Running it — plain Docker (quick local smoke test only, no Caddy)

Useful for a fast sanity check of the image itself without bringing up the whole stack — not the workflow this project actually deploys with.

```bash
# amd64 shown; swap --platform linux/arm64 for the Oracle VM
docker build -t ytpp-api .
docker run --rm -p 8000:8000 -v ./models/bundles:/app/models/bundles:ro ytpp-api

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -F thumbnail=@some_image.jpg -F title="..." \
  -F subscriber_count_at_upload=482000 -F trailing_avg_views=310000 \
  -F duration_seconds=612 -F genre=Entertainment
```

Unlike Compose, `-p 8000:8000` here does publish the port to the host directly, so plain `curl http://localhost:8000/...` works — but there's no Caddy/HTTPS in front of it, so this isn't representative of the deployed setup.

`requirements-serve.txt` is meant to be hand-curated and exactly pinned (not a `pip freeze`) to only what `serving/` imports — see the open item above, since it currently isn't quite that. torch/torchvision install from PyTorch's CPU-only wheel index in the Dockerfile (a plain install resolves to CUDA wheels the Ampere VM, no GPU, doesn't need).

### Running it — Airflow

Prereqs: `.env` filled in (see `.env.example` — now includes `AIRFLOW_UID`,
`_AIRFLOW_WWW_USER_USERNAME`/`PASSWORD` (currently unused, see note below),
`AIRFLOW_FERNET_KEY`), `docker compose up -d postgres minio` already healthy.

```bash
# 1. Build the worker image (every ingest_new/embed_new task runs inside this)
docker compose build worker

# 2. Confirm the `airflow` database exists in Postgres.
#    db/init/000_create_airflow_db.sh only runs on a FRESH postgres_data volume --
#    if postgres was already running before this file existed, create it by hand:
docker compose exec postgres psql -U $POSTGRES_USER -l   # look for "airflow" in the list
# if missing:
docker compose exec postgres psql -U $POSTGRES_USER -c "CREATE DATABASE airflow;"

# 3. Initialize Airflow's metadata DB (migration only -- no `users create`,
#    that's a FAB-only command and Airflow 3 defaults to SimpleAuthManager)
docker compose run --rm airflow-init

# 4. Start the Airflow services
docker compose up -d airflow-api-server airflow-scheduler airflow-dag-processor
docker compose ps   # all three should reach healthy/running

# 5. Get the admin login -- SimpleAuthManager auto-generates it on first
#    boot and prints it once to the api-server's logs
docker compose logs airflow-api-server | grep password
# (PowerShell: docker compose logs airflow-api-server | Select-String -Pattern "password")
# UI: http://localhost:8080, username "admin", password from that line

# --- Using Airflow UI ---
# 6. Log in with username "admin" and password based on the previous output
# 7. Execute 'run_ingestion' on 'DAG' section.

# --- Using CLI (bash/powershell) ---
# 6. Confirm both DAGs parsed with no import errors
docker compose exec airflow-scheduler airflow dags list
docker compose exec airflow-scheduler airflow dags list-import-errors

# 7. Trigger ingest_new (calls the real YouTube API -- costs real quota)
docker compose exec airflow-scheduler airflow dags test ingest_new 2026-01-01

# 8. Final checks
#   Check for MinIO content (images)
docker run --rm --network ytpp_net `                  
>>   -e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 `
>>   -e POSTGRES_USER=ytpp -e POSTGRES_PASSWORD=ytpp_2026_proj -e POSTGRES_DB=ytpp `
>>   -e MINIO_ENDPOINT=minio:9000 -e MINIO_ROOT_USER=ytpp -e MINIO_ROOT_PASSWORD=ytpp_2026_proj `
>>   ytpp-worker:latest python -m pipeline.check_consistency
#   Check for PostgreSQL content (images)
docker compose exec postgres psql -U ytpp -d ytpp -c "SELECT count(*), count(image_embedding), count(text_embedding) FROM videos;"
# Make sure that all numbers of rows match.

# 9. Training the model
docker compose exec airflow-scheduler airflow dags unpause train_model # To unpause the training phase
docker compose exec airflow-scheduler airflow dags trigger train_model # To actually run (trigger) the training phase
docker compose exec airflow-scheduler airflow dags list-runs train_model # To see if it's done.
```

**Known gaps / simplifications, not yet resolved:**
- Auth is SimpleAuthManager (dev-only, plaintext password file at
  `/opt/airflow/simple_auth_manager_passwords.json.generated` inside the
  container) — fine for local/solo use, not for anything exposed beyond
  your own machine. `_AIRFLOW_WWW_USER_USERNAME`/`PASSWORD` in `.env.example`
  are currently dead config, left over from the FAB-style setup this
  replaced.
- DB/MinIO credentials for worker tasks are read from the Airflow
  container's own env (passed through from `.env`), not from Airflow
  Connections/a secrets backend. Fine for one person on one VM, not
  beyond that.
- `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS` must be set (e.g.
  `"admin:admin"`) or no user gets created at all — SimpleAuthManager does
  **not** auto-create an `admin` user out of the box the way older Airflow
  versions did.

## Deployment (planned infra)

One Oracle Cloud Always Free Ampere A1 VM (arm64; 1 OCPU/6GB to start — resizable to 2/12 later without recreating), Docker Compose: Caddy (HTTPS) + FastAPI + Postgres + MinIO now, Airflow + worker later. No AWS unless v2 training needs a GPU spot instance. Compose file kept portable (should run on EC2 if the free tier changes).

**Planned DAGs:** `ingest_new` (daily) → `embed_new` → `finalize_labels` (~28-day horizon) → `retrain` (periodic/on drift, hard gate + paired-bootstrap soft comparison, ties go to newer) → `monitor_drift` (weekly) → `backup` (nightly, off-VM).

## Roadmap

1. ~~Drop the 7 visual tabular columns~~ — done (see Ablations).
2. ~~Export versioned bundle + parity test~~ — done.
3. ~~Slim serving reqs, arm64 Dockerfile, validate Docker Compose stack locally~~ — done (though see the open item above re: serving reqs drifting again).
4. ~~Move `videos.csv` → Postgres; containerize ingest/embed as CLI commands~~ — **done**: `pipeline/run_ingestion.py`, `models/precompute_embeddings.py`, `models/train.py` all read/write Postgres; thumbnails in MinIO. `finalize_labels` as its own step isn't split out yet — labels are still finalized inline during ingestion.
5. **Deploy to Oracle VM** — code and Compose stack validated locally (amd64); blocked on Oracle Ampere capacity, arm64 not yet tested on real hardware. Resolve the port-exposure open item before attempting this.
6. Wrap in Airflow DAGs; add `backup`.
7. Add `retrain` + promotion logic, then `monitor_drift`.
8. **v2, early fusion:** cross-attention transformer over image patches + title tokens (30-50k+ rows). Local GPU or AWS spot — undecided. Adopt only if measurably better than v1.

### Open decisions
- Retrain cadence, minimum new rows per cycle, test-slice length.
- Promotion rejection rule: strict CI-below-0, or with a margin.
- Whether `api` should actually depend on Postgres/MinIO (and if so, for what), or whether that wiring in `docker-compose.yml`/`requirements-serve.txt` should be removed.

## Repo structure

```
ingestion/     YouTube API client, thumbnail downloader (→ MinIO), subscriber lookup (TubeCensus stub)
features/      Title features, trailing views, target, visual features
pipeline/      run_ingestion.py (→ Postgres + MinIO), check_consistency.py, config/ (channels.json, cum_channels.json)
models/        precompute_embeddings.py, dataset.py, late_fusion_model.py, train.py,
               baseline.py, ablation_*.py -- read/write Postgres directly
db/init/       001_init.sql -- Postgres schema (pgvector-enabled `videos` table)
serving/       bundle.py (LoadedBundle), features.py (feature reconstruction), app.py (FastAPI)
tests/         test_bundle_parity.py
experiments/   results.jsonl, ablation_visual_flags.jsonl (one line per training/ablation run)
data/          local dataset output (gitignored) -- now just inspection artifacts (e.g. full_dataset.csv dump), not the primary store
Dockerfile, docker-compose.yml, Caddyfile, requirements-serve.txt, requirements-train.txt, .env.example
```

## Setup

```bash
pip install -r requirements-train.txt   # full dev/training freeze (CUDA torch, Jupyter, TubeCensus deps)
cp .env.example .env                    # fill in YOUTUBE_API_KEY, POSTGRES_*, MINIO_*
docker compose up -d postgres minio     # bring up the backing services before running pipeline scripts
```

Training/ingestion scripts (`pipeline/run_ingestion.py`, `models/precompute_embeddings.py`, `models/train.py`, `pipeline/check_consistency.py`) are run on the host, not inside a container — they connect to Postgres/MinIO via `localhost` using the ports Compose publishes for those two services.

## Running it

```bash
python -m pipeline.run_ingestion         # 1. ingest (incremental, safe to re-run; upserts to Postgres, thumbnails to MinIO)
python models/late_fusion_model.py       # 2. sanity-check architecture
python -m models.precompute_embeddings   # 3. precompute embeddings (reads/writes Postgres)
python -m models.train                   # 4. train (env vars: IMAGE_ENCODER, TEXT_ENCODER, USE_SIM, SEED)
python -m models.baseline                # 5. baselines / diagnostics
python -m models.ablation_modalities
python -m pipeline.check_consistency     # optional: reconcile Postgres thumbnail_path rows against MinIO bucket contents
```

Each training/ablation run appends to `experiments/results.jsonl` or `experiments/ablation_visual_flags.jsonl`; checkpoints saved to `models/checkpoints/`.