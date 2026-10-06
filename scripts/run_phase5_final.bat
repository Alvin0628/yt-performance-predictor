@echo off
REM Phase 5 - FINAL train-only dengan konfigurasi c24 (trial 24 Stage C), seed 100-104.
REM Test dibuka SEKALI per --test-role (dijaga ledger: early_fusion/results/test_eval_ledger.jsonl).
REM Jalankan dari ROOT repo (D:\PREP_INTERN\youtube_performance), venv aktif:   run_phase5_final.bat
REM
REM Langkah 1: bekukan konfigurasi ke file tersendiri (harus tercetak: trial 24)
python -c "import json;d=json.load(open('early_fusion/results/optuna/stageC_full_top3.json'));json.dump(d[0]['cfg'],open('early_fusion/results/final_cfg.json','w'),indent=2);print('trial', d[0]['trial'])"

set P=python early_fusion/experiments/run_m6_config.py
set S=--seeds 100 101 102 103 104 --save-preds --eval-test

REM Langkah 2: model final M6 (5 checkpoint disimpan untuk ensemble produksi)
%P% --config early_fusion/results/final_cfg.json %S% --test-role m6_final --save-ckpt --tag final_m6_c24

REM Langkah 3: ablasi gate (gate dibekukan di identity) pada seed yang sama; peran test terpisah
%P% --config early_fusion/results/final_cfg.json --set gate_lr_mult=0 %S% --test-role m6_gateoff --tag final_m6_c24_gateoff

echo.
echo === selesai. Kirim seluruh teks "=== ringkasan ===" dari kedua run.
