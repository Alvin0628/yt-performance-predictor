"""embed_new -- precompute CLIP embeddings for rows ingest_new added.

Runs models/precompute_embeddings.py, which is idempotent by design (only
processes rows WHERE image_embedding IS NULL OR text_embedding IS NULL), so
this is also safe to trigger manually or re-run after a failure.

schedule=None: this DAG is only ever started by ingest_new's
trigger_embed_new task, not on its own clock. Trigger it manually
(dags trigger embed_new) to backfill without a fresh ingestion run.

On completion (success OR failure -- see trigger_train_model's
trigger_rule="all_done" below), triggers train_model regardless. This is
deliberate, not an oversight: models/train.py's load_data() already
filters to `WHERE image_embedding IS NOT NULL AND text_embedding IS NOT
NULL`, so it only ever trains on rows that are actually embedded -- a
partial or failed embedding run just means this cycle's newest rows sit
out of training until a later run embeds them, not a training failure or
bad data. The only thing this doesn't protect against is a systemic
failure (e.g. Postgres/MinIO unreachable) that would also break
train_model for the same reason -- see the docstring discussion this came
out of for the reasoning.
"""
import os

import pendulum
from airflow.sdk import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

WORKER_IMAGE = "ytpp-worker:latest"
NETWORK = "ytpp_net"

WORKER_ENV = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": os.environ.get("POSTGRES_USER", ""),
    "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
    "POSTGRES_DB": os.environ.get("POSTGRES_DB", ""),
    "MINIO_ENDPOINT": "minio:9000",
    "MINIO_ROOT_USER": os.environ.get("MINIO_ROOT_USER", ""),
    "MINIO_ROOT_PASSWORD": os.environ.get("MINIO_ROOT_PASSWORD", ""),
}

with DAG(
    dag_id="embed_new",
    description="Precompute CLIP image/text embeddings for un-embedded rows",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ytpp", "embeddings"],
) as dag:
    run_embeddings = DockerOperator(
        task_id="run_embeddings",
        image=WORKER_IMAGE,
        command=["python", "-m", "models.precompute_embeddings"],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        environment=WORKER_ENV,
        auto_remove="success",
        mount_tmp_dir=False,
        execution_timeout=pendulum.duration(hours=3),
    )

    trigger_train_model = TriggerDagRunOperator(
        task_id="trigger_train_model",
        trigger_dag_id="train_model",
        wait_for_completion=False,
        # Fire regardless of run_embeddings' outcome -- the default
        # trigger_rule ("all_success") would silently skip this whenever
        # embedding fails, and train_model doesn't need it to have
        # succeeded (see the module docstring above for why).
        trigger_rule="all_done",
    )

    run_embeddings >> trigger_train_model