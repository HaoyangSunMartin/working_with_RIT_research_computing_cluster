#!/bin/bash
#
# Example: run a Hugging Face LLM on the RIT cluster with Slurm.
#
# This script demonstrates the usual pattern:
#   1. Request GPU resources from Slurm.
#   2. Enter the project directory.
#   3. Activate the host Conda/Miniforge environment.
#   4. Verify that PyTorch can see the allocated GPUs.
#   5. Run the Python LLM generation script with a Hugging Face model.
#
# Basic submit:
#   sbatch cluster/run_llm_qwen25_32b_example.slurm
#
# Submit with common overrides:
#   sbatch --export=ALL,MAX_EXAMPLES=1,EXPERIMENT_NAME=qwen_demo_1 \
#     cluster/run_llm_qwen25_32b_example.slurm
#
# Submit with a different model:
#   sbatch --export=ALL,MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct,MAX_EXAMPLES=1 \
#     cluster/run_llm_qwen25_32b_example.slurm
#
# Before using this as a real job, update:
#   - --account
#   - --partition
#   - --gres
#   - CONDA_SH
#   - CONDA_ENV_NAME
#
# Notes:
#   - Qwen2.5-32B generally needs more than one A100 40GB GPU unless quantized.
#   - HF_DEVICE_MAP=auto lets Transformers shard the model across allocated GPUs.
#   - Keep the LLM on the host Python environment; do not run it inside the
#     PostgreSQL/benchmark Apptainer container unless you intentionally need that.

#SBATCH --job-name=llm-qwen25-demo
#SBATCH --account=AccountName
#SBATCH --partition=tier3
#SBATCH --time=0-00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# 1. Move to the synced project folder on the cluster.
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/Projects/bird}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# 2. Pick the model and a small run size for the demo.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-32B-Instruct}"
MODE="${MODE:-dev}"
MAX_EXAMPLES="${MAX_EXAMPLES:-3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_32b_demo_${SLURM_JOB_ID:-manual}}"

# 3. Activate the Python environment that has torch/transformers installed.
CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-BIRD}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
PYTHON_CMD="${CONDA_PREFIX}/bin/python"

# 4. Check that Slurm actually gave this job usable GPUs.
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "hostname=$(hostname)"
nvidia-smi || true
"${PYTHON_CMD}" - <<'PY'
import sys
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    sys.stderr.write("PyTorch cannot see a CUDA GPU in this Slurm allocation.\n")
    sys.exit(2)
PY

# 5. Run the Python LLM generation script.
#    device_map=auto lets Transformers shard the model over the allocated GPUs.
HF_DEVICE_MAP="${HF_DEVICE_MAP:-auto}"
HF_TORCH_DTYPE="${HF_TORCH_DTYPE:-bfloat16}"
HF_TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE:-True}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
DB_ROOT_PATH="${DB_ROOT_PATH:-${DATA_ROOT}/${MODE}_databases/}"
EVAL_PATH="${EVAL_PATH:-${DATA_ROOT}/${MODE}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/llm/exp_result/${EXPERIMENT_NAME}/turbo_output_kg/}"
TRACE_DIR="${TRACE_DIR:-${OUTPUT_DIR}}"

mkdir -p "${OUTPUT_DIR}" "${TRACE_DIR}"

"${PYTHON_CMD}" -u llm/src/gpt_request.py \
  --backend hf \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --hf_device_map "${HF_DEVICE_MAP}" \
  --hf_torch_dtype "${HF_TORCH_DTYPE}" \
  --hf_trust_remote_code "${HF_TRUST_REMOTE_CODE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --mode "${MODE}" \
  --eval_path "${EVAL_PATH}" \
  --db_root_path "${DB_ROOT_PATH}" \
  --db_engine sqlite \
  --use_knowledge True \
  --chain_of_thought False \
  --max_examples "${MAX_EXAMPLES}" \
  --data_output_path "${OUTPUT_DIR}" \
  --trace_output_path "${TRACE_DIR}"

echo "job_complete experiment_name=${EXPERIMENT_NAME}"
