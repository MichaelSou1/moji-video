#!/usr/bin/env bash
# Generic retry-wrapper to launch a verl-omni Wan2.2 OCR eval on huirui.
# Fixes the cuDNN LD_LIBRARY_PATH pollution that broke the previous step-150 run:
#   The login shell inherited LD_LIBRARY_PATH pointing at a videorepa env's cuDNN 9.1.0,
#   which overrode the venv-bundled cuDNN 9.17.1 -> "cuDNN version incompatibility" killed
#   the reward vLLM server on every attempt. We unset it so torch uses its RPATH cudnn.
set -uo pipefail

VENV=/data/verl-omni/.venv
REPO=/data/verl-omni
SCRIPT=examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_ocr.sh

EXP=${EXP:?must set EXP}
VALDIR=${VALDIR:?must set VALDIR}
VALFILES=${VALFILES:-/data/verl-omni/workspace/data/ocr/wan22/test.parquet}
# The launch script hardcodes data.train_files=/home/user/... (dev box path) which does
# not exist here; even in val_only mode verl validates the path, so override it explicitly.
TRAINFILES=${TRAINFILES:-/data/verl-omni/workspace/data/ocr/wan22/train.parquet}
RESUME=${RESUME:-""}            # set to "auto" to load a checkpoint
EXCLUDE_GPUS=${EXCLUDE_GPUS:-""}
FIXED_GPUS=${FIXED_GPUS:-""}    # "x,y" to pin; otherwise auto-pick feasible pair
MAX_ATTEMPTS=${MAX_ATTEMPTS:-30}
RETRY_SLEEP=${RETRY_SLEEP:-180}
REWARD_UTIL=${REWARD_UTIL:-0.32}   # >=0.30 required for Qwen3-VL-8B reward KV cache
ROLLOUT_FLOOR=${ROLLOUT_FLOOR:-0.24}
CAP=${CAP:-0.92}                    # co-tenant + our reservation must stay under this
REWARD_FEASIBLE=${REWARD_FEASIBLE:-0.45}  # reward GPU neighbor must be <= this (else retry)
ROLLOUT_FEASIBLE=${ROLLOUT_FEASIBLE:-0.75} # rollout-only GPU neighbor must be <= this

cd "$REPO" || exit 1
source "$VENV/bin/activate"
# THE FIX: drop inherited LD_LIBRARY_PATH, then expose the venv's own cuDNN 9.17.1 +
# CUDA runtime explicitly. The run script will prepend its cuda-compat/cu13 dirs on top.
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"

LOG=/data/verl-omni/logs/eval_${EXP}_retry.log
echo "wrapper start $(date -u) EXP=$EXP RESUME=$RESUME" > "$LOG"

# Pick a feasible (reward_gpu, rollout_gpu) pair: lowest-neighbor GPUs that fit.
# NOTE: use a dedicated `chosen` flag, NOT the gpu index (gpu 0 would be falsy in awk).
pick_gpus() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', *' -v ex="$EXCLUDE_GPUS" -v rf="$REWARD_FEASIBLE" -v lf="$ROLLOUT_FEASIBLE" '
        BEGIN{n=split(ex,a,",");for(i=1;i<=n;i++)skip[a[i]]=1}
        !($1 in skip){print $2/$3, $1}' \
    | sort -n \
    | awk -v rf="$REWARD_FEASIBLE" -v lf="$ROLLOUT_FEASIBLE" '
        !chosen && $1<=rf { rg=$2; chosen=1; next }
        chosen && $2!=rg && $1<=lf { print rg "," $2; found=1; exit }
        END{ if(!found) print "" }'
}

for ((att=1; att<=MAX_ATTEMPTS; att++)); do
  if [ -n "$FIXED_GPUS" ]; then
    GPUS="$FIXED_GPUS"
  else
    GPUS=$(pick_gpus)
    if [ -z "$GPUS" ]; then
      echo "attempt $att: no feasible GPU pair (all too contended), sleep $RETRY_SLEEP" | tee -a "$LOG"
      sleep "$RETRY_SLEEP"; continue
    fi
  fi
  export CUDA_VISIBLE_DEVICES="$GPUS"

  # Adaptive rollout util: keep co-tenant + reward + rollout reservation under CAP.
  fmax=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', *' -v g="$GPUS" 'BEGIN{n=split(g,a,",");for(i=1;i<=n;i++)sel[a[i]]=1} $1 in sel{print $2/$3}' \
    | sort -n | tail -1)
  ROLLOUT_UTIL=$(awk "BEGIN{t=$CAP-$fmax-$REWARD_UTIL; if(t<$ROLLOUT_FLOOR)t=$ROLLOUT_FLOOR; if(t>0.40)t=0.40; printf \"%.2f\",t}")

  echo "attempt $att/$MAX_ATTEMPTS | GPU $GPUS (neighbors max=$(printf %.2f "$fmax")) | rollout=$ROLLOUT_UTIL reward=$REWARD_UTIL" | tee -a "$LOG"

  ARGS="trainer.val_only=True trainer.experiment_name=$EXP trainer.validation_data_dir=$VALDIR data.train_files=$TRAINFILES data.val_files=$VALFILES trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_UTIL reward.reward_model.rollout.gpu_memory_utilization=$REWARD_UTIL"
  [ "$RESUME" = "auto" ] && ARGS="$ARGS trainer.resume_mode=auto"

  ALOG=/data/verl-omni/logs/eval_${EXP}_retry.attempt${att}.log
  bash "$SCRIPT" $ARGS > "$ALOG" 2>&1
  rc=$?

  # The inner script pipes python|tee so its exit code is useless; judge by log content.
  if grep -qiE "cuDNN version incompatibility|No available memory for the cache blocks|out of memory|Traceback|RuntimeError" "$ALOG"; then
    echo "attempt $att FAILED (inner rc=$rc), see $ALOG; retry in $RETRY_SLEEP" | tee -a "$LOG"
    sleep "$RETRY_SLEEP"
  else
    echo "attempt $att COMPLETED (inner rc=$rc)" | tee -a "$LOG"
    break
  fi
done
