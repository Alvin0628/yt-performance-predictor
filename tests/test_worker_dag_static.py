"""Fase 6: pemeriksaan statis image worker dan DAG train_model (tanpa Airflow/Docker).

Airflow dan Docker tidak ada di mesin uji, jadi DAG dibaca lewat AST: fungsi murni (_command, _parse_decision,
_fmt) dijalankan terpisah, sisanya diperiksa sebagai teks/AST. Yang dikunci: DAG, image worker, dan CLI
retrain.py saling cocok (flag yang dipakai DAG harus diterima parser CLI).

    python -m pytest tests/test_worker_dag_static.py -q
"""
import ast
import glob
import json
import re
from pathlib import Path

import pytest

from early_fusion import live_bundle as lb
from early_fusion import retrain as rt

REPO = Path(__file__).resolve().parent.parent
DAG_SRC = (REPO / "dags/train_model.py").read_text()
DAG_TREE = ast.parse(DAG_SRC)


def _dag_funcs(*names, **env):
    """Jalankan fungsi murni dari DAG tanpa meng-import Airflow."""
    ns = {"json": json, "RETRAIN_THREADS": "1", "RETRAIN_MIN_NEW_ROWS": "100", "RETRAIN_USE_ANCHOR": True, **env}
    for node in DAG_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), "dag_src", "exec"), ns)
    return [ns[n] for n in names]


# ------------------------------------------------------------------ image worker
def test_worker_image_copies_the_retrain_code_selectively():
    lines = [l.strip() for l in (REPO / "Dockerfile.worker").read_text().splitlines() if l.strip().startswith("COPY")]
    assert not any(re.fullmatch(r"COPY early_fusion/? early_fusion/?", l) for l in lines)   # tidak menyalin seluruh pohon
    ef = [l for l in lines if "early_fusion" in l]
    assert ef, "Dockerfile.worker tidak menyalin early_fusion"
    for line in ef:
        for src in line.split()[1:-1]:                          # setiap sumber COPY harus ada di repo
            assert glob.glob(str(REPO / src)), f"COPY sumber tidak ada: {src}"
    copied = " ".join(ef)
    for needed in ("retrain.py", "promotion.py", "live_bundle.py", "data_spec.py", "early_fusion/datasets/",
                   "early_fusion/experiments/", "early_fusion/splits/", "early_fusion/models/*.py",
                   "early_fusion/results/final_cfg.json"):
        assert needed in copied, needed
    assert "COPY features/ features/" in (REPO / "Dockerfile.worker").read_text()


def test_worker_requirements_have_what_retrain_needs():
    reqs = {re.split(r"[=<>~ #]", l.strip(), maxsplit=1)[0].lower().replace("_", "-")
            for l in (REPO / "requirements-worker.txt").read_text().splitlines() if l.strip() and not l.startswith("#")}
    for pkg in ("pyarrow", "scikit-learn", "scipy", "torch", "transformers", "pandas", "numpy", "boto3", "pillow"):
        assert pkg in reqs, f"requirements-worker.txt tanpa {pkg}"


def test_every_module_retrain_imports_from_early_fusion_is_copied_into_the_image():
    """Impor level-modul di early_fusion yang dipakai retrain.py harus ada di berkas yang di-COPY."""
    copied_dirs = {"datasets", "experiments", "splits", "models"}
    root_files = {"__init__.py", "data_spec.py", "live_bundle.py", "promotion.py", "retrain.py"}
    seen = set()
    for f in [REPO / "early_fusion/retrain.py", REPO / "early_fusion/live_bundle.py", REPO / "early_fusion/promotion.py"]:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("early_fusion"):
                parts = node.module.split(".")
                if len(parts) == 1:
                    seen.update(f"{a.name}.py" for a in node.names if (REPO / "early_fusion" / f"{a.name}.py").exists())
                else:
                    seen.add(parts[1] if parts[1] in copied_dirs else f"{parts[1]}.py")
    for item in seen:
        assert item in copied_dirs or item in root_files, f"retrain mengimpor early_fusion/{item} yang tidak di-COPY"


# ------------------------------------------------------------------ DAG
def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _docker_operator_call():
    return next(n for n in ast.walk(DAG_TREE) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "DockerOperator")


def test_dag_identity_is_unchanged_so_embed_new_still_triggers_it():
    assert 'dag_id="train_model"' in DAG_SRC and "schedule=None" in DAG_SRC and "max_active_runs=1" in DAG_SRC
    assert "def _restart_api(" in DAG_SRC and 'com.docker.compose.service=api' in DAG_SRC


def test_docker_operator_pushes_all_lines_and_has_a_timeout():
    call = _docker_operator_call()
    assert ast.literal_eval(_kw(call, "xcom_all")) is True and ast.literal_eval(_kw(call, "do_xcom_push")) is True
    assert _kw(call, "execution_timeout") is not None and "RETRAIN_TIMEOUT_HOURS" in DAG_SRC


def test_worker_environment_and_mounts_cover_what_retraining_touches():
    for key in ("POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
                "MINIO_ENDPOINT", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD"):
        assert f'"{key}"' in DAG_SRC, key
    targets = set(re.findall(r'target="(/app/[^"]+)"', DAG_SRC))
    assert targets == {"/app/models/bundles", "/app/data_snapshots", "/app/experiments"}
    for t in targets:                                            # sumber host = HOST_PROJECT_DIR/<folder yang sama>
        rel = t.removeprefix("/app/")
        assert 'source=f"{HOST_PROJECT_DIR}/' + rel + '"' in DAG_SRC


def test_command_uses_the_stable_bundle_names_and_flags_the_cli_accepts():
    (command,) = _dag_funcs("_command")
    cmd = command()
    assert cmd[:5] == ["python", "-m", "early_fusion.retrain", "all", "--root"] and "--promote" in cmd
    assert cmd[cmd.index("--live") + 1] == f"/app/models/bundles/{lb.LATEST}.pt"
    assert cmd[cmd.index("--anchor") + 1] == f"/app/models/bundles/{lb.ANCHOR}.pt"
    assert cmd[cmd.index("--bundles-dir") + 1] == "/app/models/bundles"
    assert cmd[cmd.index("--log-path") + 1] == "/app/experiments/promotions.jsonl"
    assert cmd[cmd.index("--threads") + 1] == "1" and cmd[cmd.index("--min-new-rows") + 1] == "100"
    # DAG <-> CLI: semua argumen setelah "all" harus diterima parser retrain.py
    ns = rt.build_parser().parse_args(["all"] + cmd[cmd.index("all") + 1:])
    assert ns.promote is True and ns.anchor.endswith("m6_anchor.pt") and ns.live.endswith("m6_latest.pt")
    assert ns.min_new_rows == 100 and ns.threads == 1
    # --anchor bisa dimatikan; semua string (DockerOperator tidak menerima angka di command)
    assert "--anchor" not in command(use_anchor=False) and all(isinstance(x, str) for x in cmd)


def test_check_promotion_restarts_the_api_only_when_the_bundle_was_applied():
    assert "decision.get(\"promoted\") and decision.get(\"applied\")" in DAG_SRC
    body = DAG_SRC.split("def check_promotion", 1)[1]
    assert body.index("_restart_api(logger)") < body.index("elif decision.get(\"promoted\")")   # restart hanya di cabang applied
    assert body.count("_restart_api(logger)") == 1


def test_parse_decision_finds_the_json_line_even_with_noise_around_it():
    parse, = _dag_funcs("_parse_decision")
    dec = {"promoted": True, "applied": True, "new_spearman": 0.44}
    line = json.dumps(dec)
    assert parse(line) == dec                                              # satu baris (xcom_all=False)
    assert parse(["Loading weights: 100%", "{not json}", line, "Warning: late shutdown message"]) == dec
    assert parse([line + "\n"]) == dec and parse(line.encode()) == dec
    assert parse("log awal\n" + line + "\nlog akhir") == dec               # satu string multi-baris
    assert parse(dec) == dec and parse({"x": 1}) is None
    # JSON lain tanpa kunci "promoted" diabaikan, yang terakhir menang
    older = json.dumps({"promoted": False})
    assert parse([older, '{"other": 1}', line]) == dec and parse([line, older]) == {"promoted": False}
    assert parse([]) is None and parse(["tidak ada json"]) is None and parse(None) is None and parse(5) is None


def test_fmt_handles_missing_spearman_for_skipped_runs():
    fmt, = _dag_funcs("_fmt")
    assert fmt(None) == "n/a" and fmt(0.43130959) == "0.4313"
