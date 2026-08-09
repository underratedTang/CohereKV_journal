#!/usr/bin/env bash
set -euo pipefail

# =========================
# 手动修改区
# =========================

ROOT="/home/tlxr/SimCalKV"
TASK_DIR="/home/tlxr/SimCalKV/task_evaluation"
CUDA_DIR="/home/tlxr/SimCalKV/mymethod_cuda_src"
OUT_DIR="/home/tlxr/SimCalKV/results/mymethod_engineering_validation"

# 模型设置
FAMILY="llama"                       # llama | qwen
MODEL="meta-llama/Llama-3.2-3B"
# FAMILY="qwen"
# MODEL="Qwen/Qwen2.5-7B"

# 运行设置
DEVICE_MAP="auto"
DEVICE="cuda"
DTYPE="bfloat16"
CAPTURE_LENGTHS="2048 4096 8192"
# Snapshot attention-score capture is exact but query-blocked to control peak VRAM.
# RTX 4090 (24 GB): 64 is the conservative default; 128 can be tried after validation.
CAPTURE_ATTN_CHUNK_SIZE=64
LAMBDA_LIST="0.5,0.75"
COMPRESS_RATIO=0.4
WARMUP=3
REPEAT=10

# PPL 仅用于 Torch/CUDA backend 数值一致性，不用于比较 selector 优劣
RUN_PPL_PARITY=1
PPL_SEQLEN=4096
PPL_MAX_SEQUENCES=1
PPL_CHUNK_SIZE=32
PPL_LAMBDA=0.75

# 环境设置：直接在这里修改，不使用 MODEL=... bash xxx.sh 形式覆盖
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

# =========================
# 自动区：一般不用改
# =========================
mkdir -p "${OUT_DIR}"
cd "${ROOT}"

{
  date -Is
  nvidia-smi -L || true
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || true
  python - <<'PY'
import torch,sys
print('python',sys.version.replace('\n',' '))
print('torch',torch.__version__,'cuda',torch.version.cuda,'cuda_available',torch.cuda.is_available())
try:
 import mymethod_cuda
 print('mymethod_cuda_available',mymethod_cuda.is_available(),'error',repr(mymethod_cuda.extension_error()))
except Exception as e: print('mymethod_cuda_import_error',repr(e))
PY
} 2>&1 | tee "${OUT_DIR}/environment.txt"

# Hard correctness gates before any expensive model evaluation.
python "${TASK_DIR}/verify_mymethod_decoupling.py" 2>&1 | tee "${OUT_DIR}/decoupling_check.log"
python "${CUDA_DIR}/test_mymethod_cuda.py" 2>&1 | tee "${OUT_DIR}/cuda_extension_smoke.log"
(cd "${TASK_DIR}" && python -m unittest -q \
  tests.test_mymethod \
  tests.test_ablation_wrapper \
  tests.test_centroidkv) 2>&1 | tee "${OUT_DIR}/unittest.log"

for L in ${CAPTURE_LENGTHS}; do
  SNAP="${OUT_DIR}/kv_snapshot_${L}.pt"
  if [[ ! -f "${SNAP}" ]]; then
    python "${TASK_DIR}/capture_mymethod_kv_snapshot.py" \
      --model_name "${MODEL}" --family "${FAMILY}" --dataset wikitext2 \
      --model_seqlen "${L}" --sample_index 0 --layers first,mid,last \
      --torch_dtype "${DTYPE}" --device_map "${DEVICE_MAP}" --attn_implementation eager \
      --attention_chunk_size "${CAPTURE_ATTN_CHUNK_SIZE}" \
      --output "${SNAP}" 2>&1 | tee "${OUT_DIR}/capture_${L}.log"
  fi
  python "${TASK_DIR}/validate_mymethod_cuda_backend.py" \
    --snapshot "${SNAP}" --lambda_mymethod 0.75 --compress_ratio "${COMPRESS_RATIO}" \
    --output_json "${OUT_DIR}/backend_validation_${L}.json" \
    2>&1 | tee "${OUT_DIR}/backend_validation_${L}.log"
  python "${TASK_DIR}/benchmark_mymethod_engineering.py" \
    --snapshot "${SNAP}" --lambdas "${LAMBDA_LIST}" --compress_ratio "${COMPRESS_RATIO}" \
    --device "${DEVICE}" --warmup "${WARMUP}" --repeat "${REPEAT}" \
    --output_json "${OUT_DIR}/engineering_benchmark_${L}.json" \
    2>&1 | tee "${OUT_DIR}/engineering_benchmark_${L}.log"
done

run_ppl_parity() {
  local selector="$1" route="$2" graph="$3" tag="$4"
  python "${TASK_DIR}/evaluate_ppl_kvmerge.py" \
    --model_name "${MODEL}" --family "${FAMILY}" --task_eval_dir "${TASK_DIR}" \
    --datasets wikitext2 --kv_method mymethod --compress_ratio "${COMPRESS_RATIO}" \
    --lambda_mymethod "${PPL_LAMBDA}" --mymethod_selector "${selector}" --mymethod_group_mode full \
    --mymethod_route_backend "${route}" --mymethod_graph_backend "${graph}" \
    --mymethod_distance_chunk_size 256 --mymethod_attention_chunk_size 128 \
    --model_seqlen "${PPL_SEQLEN}" --chunk_size "${PPL_CHUNK_SIZE}" --compress_every 1 \
    --max_eval_sequences "${PPL_MAX_SEQUENCES}" --torch_dtype "${DTYPE}" --amp_dtype none \
    --device "${DEVICE}" --device_map "${DEVICE_MAP}" --attn_implementation eager \
    --output_json "${OUT_DIR}/ppl_parity_${tag}.json" \
    2>&1 | tee "${OUT_DIR}/ppl_parity_${tag}.log"
}

if [[ "${RUN_PPL_PARITY}" == "1" ]]; then
  # Backend equivalence only: do not interpret selector ranking from these tiny runs.
  run_ppl_parity attention torch torch attention_torch
  run_ppl_parity attention cuda cuda attention_cuda
  run_ppl_parity keepkv torch torch keepkv_torch
  run_ppl_parity keepkv cuda cuda keepkv_cuda
  python "${TASK_DIR}/check_mymethod_ppl_backend_parity.py" \
    --attention_torch "${OUT_DIR}/ppl_parity_attention_torch.json" \
    --attention_cuda "${OUT_DIR}/ppl_parity_attention_cuda.json" \
    --keepkv_torch "${OUT_DIR}/ppl_parity_keepkv_torch.json" \
    --keepkv_cuda "${OUT_DIR}/ppl_parity_keepkv_cuda.json" \
    --ppl_abs_tol 0.005 --output_json "${OUT_DIR}/ppl_backend_parity_check.json" \
    2>&1 | tee "${OUT_DIR}/ppl_backend_parity_check.log"
fi

python "${TASK_DIR}/summarize_mymethod_engineering.py" \
  --result_dir "${OUT_DIR}" --output_json "${OUT_DIR}/engineering_summary.json" \
  --output_md "${OUT_DIR}/engineering_summary.md" \
  2>&1 | tee "${OUT_DIR}/engineering_summary.log"

bash "${TASK_DIR}/package_mymethod_engineering_feedback.sh" "${OUT_DIR}" | tee "${OUT_DIR}/feedback_zip_path.txt"
