"""monitor_drift -- reports whether recently-matured videos' tabular
feature distributions have shifted vs. the rest of the finalized history.

schedule=None: started only by ingest_new's trigger_monitor_drift task, in
parallel with embed_new (a sibling, not a dependency -- see
pipeline/monitor_drift.py's docstring for why that's safe: this only
touches columns run_ingestion.py fills in directly, never embeddings).
Purely informational -- writes to experiments/drift_reports.jsonl and
never fails on a detected shift, only on an actual error.
"""
import os

import pendulum
from airflow.sdk import DAG
from airflow.providers.docker.operators.docker import DockerOperator

WORKER_IMAGE = "ytpp-worker:latest"
NETWORK = "ytpp_net"

WORKER_ENV = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": os.environ.get("POSTGRES_USER", ""),
    "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
    "POSTGRES_DB": os.environ.get("POSTGRES_DB", ""),
}

with DAG(
    dag_id="monitor_drift",
    description="Report tabular feature drift in recently-matured videos",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ytpp", "monitoring"],
) as dag:
    DockerOperator(
        task_id="run_drift_check",
        image=WORKER_IMAGE,
        command=["python", "-m", "pipeline.monitor_drift"],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        environment=WORKER_ENV,
        auto_remove="success",
        mount_tmp_dir=False,
        execution_timeout=pendulum.duration(minutes=30),
    )