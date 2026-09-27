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
from docker.types import Mount

WORKER_IMAGE = "ytpp-worker:latest"
NETWORK = "ytpp_net"

# Fail at DAG-parse time with a clear message rather than at task-run time
# with a confusing Docker mount error, if this hasn't been set up yet.
HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]

WORKER_ENV = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": os.environ.get("POSTGRES_USER", ""),
    "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
    "POSTGRES_DB": os.environ.get("POSTGRES_DB", ""),
}

# Same reason train_model.py needs this: auto_remove="success" wipes
# anything written only inside the container's own filesystem, and
# pipeline/monitor_drift.py writes its report to the local relative path
# experiments/drift_reports.jsonl (-> /app/experiments in the worker image).
# Without bind-mounting that back to the host, the write succeeds inside the
# container, then disappears with it -- no task failure, just a report that
# never shows up on disk.
MOUNTS = [
    Mount(source=f"{HOST_PROJECT_DIR}/experiments", target="/app/experiments", type="bind"),
]

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
        mounts=MOUNTS,
        auto_remove="success",
        mount_tmp_dir=False,
        execution_timeout=pendulum.duration(minutes=30),
    )