@echo off
REM Phase 4 - konfirmasi di seed segar 44 45 46 (VAL saja, tanpa --eval-test)
REM Jalankan dari ROOT repo (D:\PREP_INTERN\youtube_performance) dengan venv aktif:
REM     run_phase4_confirm.bat
REM Aman dihentikan (Ctrl+C) dan dijalankan ulang: tiap tag selesai ditulis ke m6_final_runs.jsonl.
REM Urutan top3 Stage C: :0 = trial 24 | :1 = trial 0 (pemenang Stage B) | :2 = trial 6

set O=early_fusion/results/optuna
set P=python early_fusion/experiments/run_m6_config.py
set S=--seeds 44 45 46 --save-preds

%P% --config %O%/stageC_full_top3.json:0 %S% --tag confirm_c24
%P% --config %O%/stageC_full_top3.json:1 %S% --tag confirm_c0_stageB
%P% --config %O%/stageC_full_top3.json:2 %S% --tag confirm_c6
%P% --config %O%/stageA_full_best.json %S% --tag confirm_stageA_t31
%P% %S% --tag confirm_default
%P% --config %O%/stageC_full_top3.json:0 --set gate_lr_mult=0 %S% --tag confirm_c24_gateoff

echo.
echo === selesai. Lanjut: python early_fusion/experiments/analyze_confirm.py --baseline confirm_default --candidates confirm_c24 confirm_c0_stageB confirm_c6 confirm_stageA_t31 --extra confirm_c24_gateoff --gate-pair confirm_c24 confirm_c24_gateoff
