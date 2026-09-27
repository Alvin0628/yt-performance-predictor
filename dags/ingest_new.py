"""ingest_new -- monthly incremental ingestion.

Runs pipeline/run_ingestion.py (already idempotent/upsert-safe -- see its
docstring) inside a ytpp-worker container on the same Docker network as
postgres/minio, so it can reach them by service name.

Build the worker image before this DAG can run:
    docker compose build worker

Known simplification: DB/MinIO credentials are read from this container's
own environment (passed through from .env via docker-compose.yml's
x-airflow-common env_file). For anything beyond single-user local/VM
deployment, move these to Airflow Connections/Variables or a secrets
backend instead.
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
    "YOUTUBE_API_KEY": os.environ.get("YOUTUBE_API_KEY", ""),
}

with DAG(
    dag_id="ingest_new",
    description="Incremental YouTube ingestion -> Postgres + MinIO",
    schedule="@monthly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ytpp", "ingestion"],
) as dag:
    run_ingestion = DockerOperator(
        task_id="run_ingestion",
        image=WORKER_IMAGE,
        command=["python", "-m", "pipeline.run_ingestion"],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        environment=WORKER_ENV,
        auto_remove="success",
        mount_tmp_dir=False,
        # Ingestion hits the YouTube API and downloads thumbnails -- give it
        # real headroom rather than Airflow's default task timeout.
        execution_timeout=pendulum.duration(hours=2),
    )

    trigger_embed_new = TriggerDagRunOperator(
        task_id="trigger_embed_new",
        trigger_dag_id="embed_new",
        wait_for_completion=False,
    )

    trigger_monitor_drift = TriggerDagRunOperator(
        task_id="trigger_monitor_drift",
        trigger_dag_id="monitor_drift",
        wait_for_completion=False,
    )

    run_ingestion >> [trigger_embed_new, trigger_monitor_drift]