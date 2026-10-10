"""train_model -- retrain the early-fusion (RATF M6) ensemble and swap it in if the promotion gate agrees.

Runs `python -m early_fusion.retrain all` inside a ytpp-worker container: export a fresh snapshot from
Postgres -> update the token store (only new videos) -> split -> refit 5 seeds on train+val -> package ->
verify -> gate (candidate vs the live bundle, plus the fixed anchor) -> if promoted, replace
models/bundles/m6_latest.* (the previous live bundle is kept as m6_prev.*). The container prints a JSON
decision as its last line; check_promotion below reads it and restarts the `api` container only when the
bundle was actually replaced (`applied`). Decisions are appended to experiments/promotions.jsonl.

schedule=None: not on its own clock. Triggered by embed_new's trigger_train_model task (trigger_rule
"all_done"). Still runnable by hand (`airflow dags trigger train_model`). KEEP IT PAUSED until the
Phase 8 checklist (benchmark on the VM, bootstrap of m6_latest/m6_anchor, token store in place) is done.

What must exist on the host before the first run (HOST_PROJECT_DIR is the repo as the *Docker daemon* sees it):
  models/bundles/m6_latest.pt + .json and m6_anchor.pt + .json
      python -m early_fusion.live_bundle bootstrap --src <current live bundle> --train-end ... --snapshot-n-total ...
  data_snapshots/token_store/   (copy of the token cache, or `retrain tokens --create-empty` to extract everything)
  data_snapshots/retrain/       (runs are written here; prune old ones by hand for now)
  experiments/                  (promotions.jsonl is appended here)

Knobs (environment of the Airflow containers; defaults are conservative):
  RETRAIN_THREADS         torch CPU threads for the worker (default 1: leave cores for the API)
  RETRAIN_MIN_NEW_ROWS    skip retraining when fewer new videos than this since the live bundle (default 100)
  RETRAIN_TIMEOUT_HOURS   task timeout (default 12)
  RETRAIN_USE_ANCHOR      "0" disables the anchor gate (default on)

XCom: xcom_all=True hands back every line of container output; the decision is the last line that is a JSON
object with a "promoted" key, so a stray warning printed after it cannot break the parse.
A rejected or skipped retrain is a normal outcome, not a task failure. A promotion that was decided but could
not be applied makes the container exit non-zero, so the task fails visibly (and the API is not restarted).
"""
import json
import logging
import os

import pendulum
from airflow.sdk import DAG, task
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

WORKER_IMAGE = "ytpp-worker:latest"
NETWORK = "ytpp_net"

# Fail at DAG-parse time with a clear message rather than at task-run time
# with a confusing Docker mount error, if this hasn't been set up yet.
HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]

RETRAIN_THREADS = os.environ.get("RETRAIN_THREADS", "1")
RETRAIN_MIN_NEW_ROWS = os.environ.get("RETRAIN_MIN_NEW_ROWS", "100")
RETRAIN_TIMEOUT_HOURS = int(os.environ.get("RETRAIN_TIMEOUT_HOURS", "12"))
RETRAIN_USE_ANCHOR = os.environ.get("RETRAIN_USE_ANCHOR", "1") != "0"

WORKER_ENV = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": os.environ.get("POSTGRES_USER", ""),
    "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
    "POSTGRES_DB": os.environ.get("POSTGRES_DB", ""),
    # token store updates read thumbnails from MinIO
    "MINIO_ENDPOINT": "minio:9000",
    "MINIO_ROOT_USER": os.environ.get("MINIO_ROOT_USER", ""),
    "MINIO_ROOT_PASSWORD": os.environ.get("MINIO_ROOT_PASSWORD", ""),
}

MOUNTS = [
    # live/prev/anchor bundles; the same host directory the `api` service mounts read-only
    Mount(source=f"{HOST_PROJECT_DIR}/models/bundles", target="/app/models/bundles", type="bind"),
    # runs (snapshot, checkpoints, candidate bundle) and the shared token store; must survive the container
    Mount(source=f"{HOST_PROJECT_DIR}/data_snapshots", target="/app/data_snapshots", type="bind"),
    Mount(source=f"{HOST_PROJECT_DIR}/experiments", target="/app/experiments", type="bind"),
]


def _command(threads=RETRAIN_THREADS, min_new_rows=RETRAIN_MIN_NEW_ROWS, use_anchor=RETRAIN_USE_ANCHOR):
    cmd = [
        "python", "-m", "early_fusion.retrain", "all",
        "--root", "/app/data_snapshots/retrain",
        "--store", "/app/data_snapshots/token_store",
        "--live", "/app/models/bundles/m6_latest.pt",
        "--bundles-dir", "/app/models/bundles",
        "--log-path", "/app/experiments/promotions.jsonl",
        "--promote",
        "--min-new-rows", str(min_new_rows),
        "--threads", str(threads),
    ]
    if use_anchor:
        cmd += ["--anchor", "/app/models/bundles/m6_anchor.pt"]
    return cmd


def _parse_decision(xcom):
    """Find the promotion decision in run_training's XCom.

    Accepts a dict, a string (one or more lines) or a list of lines (xcom_all=True). Scans from the end for
    the last line that is a JSON object with a "promoted" key; returns None if there is none.
    """
    if isinstance(xcom, dict):
        return xcom if "promoted" in xcom else None
    if isinstance(xcom, (bytes, str)):
        xcom = [xcom]
    if not isinstance(xcom, (list, tuple)):
        return None
    lines = []
    for raw in xcom:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        lines.extend(text.splitlines())
    for line in reversed(lines):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "promoted" in obj:
            return obj
    return None


def _fmt(x):
    return "n/a" if x is None else f"{float(x):.4f}"


def _restart_api(logger):
    """Restart the api container so it loads the newly promoted bundle.
    Found by its compose label, so it doesn't depend on the project name."""
    import docker

    client = docker.from_env()
    containers = client.containers.list(
        filters={"label": "com.docker.compose.service=api"}
    )
    if not containers:
        raise RuntimeError("no running container with compose service label 'api'")
    for c in containers:
        logger.info("Restarting %s so it loads the promoted bundle", c.name)
        c.restart(timeout=30)


with DAG(
    dag_id="train_model",
    description="Retrain the early-fusion ensemble; promote it if the gate passes",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ytpp", "training", "early-fusion"],
) as dag:
    run_training = DockerOperator(
        task_id="run_training",
        image=WORKER_IMAGE,
        command=_command(),
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        environment=WORKER_ENV,
        mounts=MOUNTS,
        auto_remove="success",
        mount_tmp_dir=False,
        # 5 seeds x 8 epochs; the real duration on the VM is measured in Phase 8 (CPU-bound there).
        execution_timeout=pendulum.duration(hours=RETRAIN_TIMEOUT_HOURS),
        # xcom_all=True: every output line, so check_promotion can pick the JSON decision even if something
        # was printed after it (stderr and stdout are merged in the container log).
        do_xcom_push=True,
        xcom_all=True,
    )

    @task
    def check_promotion(promotion_output):
        logger = logging.getLogger("train_model.promotion")
        decision = _parse_decision(promotion_output)
        if decision is None:
            logger.warning("No promotion decision (JSON line with a 'promoted' key) in run_training's output.")
            return
        if decision.get("skipped"):
            logger.info("SKIPPED -- %s", decision.get("reason"))
            return
        if decision.get("promoted") and decision.get("applied"):
            logger.info("PROMOTED AND APPLIED %s -- %s (new_spearman=%s)",
                        decision.get("applied_latest") or decision.get("versioned_path"),
                        decision.get("reason"), _fmt(decision.get("new_spearman")))
            _restart_api(logger)
        elif decision.get("promoted"):
            # Decided but not applied (no --promote): nothing changed on disk, so there is nothing to reload.
            logger.warning("PROMOTE decided but NOT applied -- API not restarted (%s)", decision.get("reason"))
        else:
            # Deliberately a warning, not a raised exception -- a rejected promotion is expected behaviour.
            logger.warning("NOT PROMOTED %s -- %s (new_spearman=%s)",
                           decision.get("versioned_path"), decision.get("reason"), _fmt(decision.get("new_spearman")))

    # No explicit `>>` needed -- passing run_training.output as an argument
    # already makes check_promotion depend on run_training.
    check_promotion(run_training.output)
