#!/usr/bin/env bash
set -u
if [ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]; then exit 0; fi
mode=${1:-cpu}
category=${2:-Industrial_and_Scientific}
run_id="$(date -u +%Y%m%dT%H%M%SZ)-${mode}-${category}-$$"
export RECSYS_RESULTS="/home/jovyan/xandi281/recsys-validation/$run_id"
export RECSYS_WORK="/tmp/recsys-$run_id"
mkdir -p "$RECSYS_RESULTS" "$RECSYS_WORK"
exec > >(tee "$RECSYS_RESULTS/run.log") 2>&1
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 USE_TF=0 USE_FLAX=0
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME="$RECSYS_WORK/hf" HF_DATASETS_CACHE="$RECSYS_WORK/datasets"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
export PYTHONPATH="$PWD"
echo "RESULT_DIR=$RECSYS_RESULTS"
df -h /tmp /home/jovyan
df -i /home/jovyan
python -m venv --system-site-packages "$RECSYS_WORK/venv"
code=$?
if [ "$code" = 0 ]; then
  export PATH="$RECSYS_WORK/venv/bin:$PATH"
  python -m pip install --no-cache-dir fire==0.7.1
  code=$?
fi
if [ "$code" = 0 ]; then
  timeout 1800s python -u cloud/smoke.py "$mode" "$category"
  code=$?
fi
echo "RECSYS_EXIT=$code"
echo "$code" > "$RECSYS_RESULTS/exit-code.txt"
exit 0
