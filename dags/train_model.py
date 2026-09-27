"""train_model -- retrain the late-fusion model on all currently embedded rows.

Runs models/train.py inside a ytpp-worker container.

Unlike ingest_new/embed_new, this task's outputs must survive the container's
removal: auto_remove="success" wipes anything written only inside the
container's own filesystem, and train.py writes its checkpoint, model bundle,
results log and plots to local paths (models/checkpoints, models/bundles,
experiments/results.jsonl, models/plots). So this DAG bind-mounts those paths
back to the host. Writing into the same host directory the `api` service
already mounts read-only (./models/bundles) means a freshly trained bundle is
picked up without restarting the API -- just re-point MODEL_BUNDLE_PATH or
restart `api` to load it, depending on how serving/bundle.py picks the file.

Requires HOST_PROJECT_DIR in .env: the path to this repo as the *Docker
daemon* sees it (reached via /var/run/docker.sock), not the path inside the
scheduler container. On Docker Desktop for Windows this is the same path you
use for Docker Desktop's file sharing, e.g.
D:/Material/Programming/Machine Learning/yt-performance-predictor/yt-performance-predictor
Forward slashes, even on Windows.

Before the first run, make sure the target directories exist on the host:
    mkdir -p models/bundles models/plots
(models/checkpoints and experiments already exist in this repo.)

Promotion visibility: models/train.py's promote_if_better() always decides
whether the new bundle actually replaces the served one (see its own
docstring), win or lose, and logs that decision to
experiments/promotions.jsonl regardless. That file isn't visible from the
Airflow UI though, so run_training also pushes it as this task's XCom via
plain do_xcom_push -- train.py's _write_airflow_xcom() prints the decision
as the very last line of stdout, which DockerOperator returns as-is (a
JSON string) and check_promotion below parses and surfaces as a task log
line. (An earlier version of this tried retrieve_output/get_archive +
pickle instead; that silently produced no XCom at all in practice, so this
went back to the simpler, already-working stdout-tail approach.) A rejected
promotion is treated as a normal outcome here, not a task failure -- it's
expected behavior (a worse retrain, or one that ties and correctly deferred
to the still-current model), not something broken that should page anyone
or trigger a retry. For now this only logs; swap the body of check_promotion
for a real notification (Slack/email/etc.) once there's a logging/alerting
module to call instead.
"""
import logging
import os
import json

import pendulum
from airflow.sdk import DAG, task
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

MOUNTS = [
    Mount(source=f"{HOST_PROJECT_DIR}/models/checkpoints", target="/app/models/checkpoints", type="bind"),
    Mount(source=f"{HOST_PROJECT_DIR}/models/bundles", target="/app/models/bundles", type="bind"),
    Mount(source=f"{HOST_PROJECT_DIR}/models/plots", target="/app/models/plots", type="bind"),
    Mount(source=f"{HOST_PROJECT_DIR}/experiments", target="/app/experiments", type="bind"),
]

with DAG(
    dag_id="train_model",
    description="Retrain the late-fusion model on all embedded rows",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ytpp", "training"],
) as dag:
    run_training = DockerOperator(
        task_id="run_training",
        image=WORKER_IMAGE,
        command=["python", "-m", "models.train"],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        environment=WORKER_ENV,
        mounts=MOUNTS,
        auto_remove="success",
        mount_tmp_dir=False,
        # 200 epochs w/ early stopping (patience 10), CPU-bound -- give it
        # real headroom. Tighten once you know how long a real run takes.
        execution_timeout=pendulum.duration(hours=6),
        # Second correction here too, for the record: tried
        # retrieve_output=True + retrieve_output_path (Docker get_archive +
        # unpickle) first as the "structured" option, but it silently
        # produced no XCom at all in practice (DockerOperator swallows any
        # APIError from that call with no logging). Reverted to the simpler,
        # already-proven mechanism: do_xcom_push (xcom_all=False, default)
        # just returns the last non-empty line of container stdout, which
        # is exactly what worked in the first two real runs. train.py's
        # _write_airflow_xcom() prints a single JSON line as the very last
        # thing main() does, specifically so that line is what lands here.
        do_xcom_push=True,
    )

    @task
    def check_promotion(promotion_decision):
        logger = logging.getLogger("train_model.promotion")

        # do_xcom_push hands back a plain string (the container's last
        # stdout line) -- not a parsed object -- so decode it here.
        if isinstance(promotion_decision, str):
            try:
                promotion_decision = json.loads(promotion_decision)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Could not parse promotion decision XCom as JSON: %r",
                    promotion_decision,
                )
                return

        if not isinstance(promotion_decision, dict):
            logger.warning(
                "No usable promotion decision from run_training's XCom (got %r).",
                promotion_decision,
            )
            return
        if promotion_decision.get("promoted"):
            logger.info(
                "PROMOTED %s -- %s (new_spearman=%.4f)",
                promotion_decision.get("versioned_path"),
                promotion_decision.get("reason"),
                promotion_decision.get("new_spearman", float("nan")),
            )
        else:
            # Deliberately a warning, not a raised exception -- a rejected
            # promotion is expected behavior, not a task failure. Raising here
            # would trigger retries/alerting meant for actual breakage.
            logger.warning(
                "NOT PROMOTED %s -- %s (new_spearman=%.4f)",
                promotion_decision.get("versioned_path"),
                promotion_decision.get("reason"),
                promotion_decision.get("new_spearman", float("nan")),
            )

    # No explicit `>>` needed -- passing run_training.output as an argument
    # already makes check_promotion depend on run_training.
    check_promotion(run_training.output)