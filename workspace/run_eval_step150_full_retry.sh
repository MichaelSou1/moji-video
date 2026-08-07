#!/usr/bin/env bash
# Auto-retry wrapper for the step-150 full evaluation (1018-row test.parquet).
#
# Why this is non-trivial: the machine (pku-tangh / huirui) is heavily
# multi-tenant. Every non-baseline GPU already has a co-tenant occupying
# 40-60% of VRAM, so a 2-GPU eval (Wan2.2-5B rollout + Qwen3-VL-8B reward) is
# fragile:
#   * rollout gpu_memory_utilization too LOW  -> vLLM "No available memory for
#     the cache blocks" (engine fails to start).
#   * rollout gpu_memory_utilization too HIGH -> CUDA OOM when the co-tenant
#     balloons during generation.
# We resolve this by (a) re-picking the two lowest-occupied GPUs every attempt,
# (b) setting rollout_util ADAPTIVELY so co-tenant + our reservation stays <=~90%,
# and (c) auto-raising the rollout floor if a KV-cache error is seen, then
# retrying many times to catch a moment when both GPUs have headroom.
#
# IMPORTANT: resume_mode=auto + experiment_name=wan22_5b_t2v_ocr_s150 makes verl
# load checkpoints/dance_grpo/wan22_5b_t2v_ocr_s150/global_step_150. Do NOT drop
# resume_mode=auto or the eval would run on a randomly-initialized model.
#
# NOTE: run_wan22_5b_t2v_ocr.sh pipes `python | tee`, so its exit code is tee's
# (always 0). We detect real failure by grepping the attempt log for tracebacks /
# OOM / KV-cache errors instead of trusting $?.

set -o pipefail

REPO=/data/verl-omni
cd "$REPO" || exit 1

# Replicate the baseline run's environment (verl-omni venv, python3.12).
# Without this, `python3` resolves to the system 3.10 which lacks `diffusers`.
source /data/verl-omni/.venv/bin/activate

export WORKSPACE=${WORKSPACE:-$REPO/workspace}

MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
EXCLUDE_GPUS=${EXCLUDE_GPUS:-3,6}
# reward model (Qwen3-VL-8B, max_seq_len=16384) needs >=~0.30 util: at 0.22 the 8B
# weights ate the whole budget and left only 1 GiB KV -> "No available memory for
# the cache blocks". 0.30*97GB ~= 29GB leaves ~13GB KV, comfortably above the 2.25GiB needed.
REWARD_UTIL=${REWARD_UTIL:-0.30}
RETRY_SLEEP=${RETRY_SLEEP:-150}
ROLLOUT_FLOOR=${ROLLOUT_FLOOR:-0.28}  # >=0.24 works for rollout KV; keep margin above it
CAP=${CAP:-0.96}                       # co-tenant + our total reservation must stay <= this

MASTER_LOG=$REPO/logs/eval_step150_full_retry.log

# Two GPUs with the LOWEST co-tenant occupancy (baseline GPUs excluded), but only
# among GPUs that can actually fit our reservation (frac + reward + rollout_floor
# must stay under CAP). This avoids picking a 2nd GPU that is too contended to ever
# fit, which would just OOM. Falls back to the two least-occupied overall if fewer
# than two feasible GPUs exist.
pick_gpus() {
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', *' -v ex="$EXCLUDE_GPUS" -v cap="$CAP" -v rw="$REWARD_UTIL" -v rf="$ROLLOUT_FLOOR" '
            BEGIN { n = split(ex, a, ","); for (i = 1; i <= n; i++) skip[a[i]] = 1 }
            !($1 in skip) {
                frac = $2 / $3
                print (frac + rw + rf <= cap) ? 0 : 1, frac, $1
            }
          ' \
        | sort -k1,1n -k2,2n \
        | head -2 \
        | awk '{print $3}' \
        | paste -sd, -
}

gpu_frac() {
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', *' -v g="$1" '$1 == g { printf "%.3f", $2 / $3 }'
}

log_fatal() {
    grep -qiE "Traceback \(most recent call last\)|ModuleNotFoundError|OutOfMemoryError|CUDA out of memory|No available memory for the cache blocks" "$1"
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    GPUS=$(pick_gpus)
    g1=${GPUS%,*}; g2=${GPUS#*,}
    f1=$(gpu_frac "$g1"); f2=$(gpu_frac "$g2")
    fmax=$(awk "BEGIN{ if ($f1 > $f2) print $f1; else print $f2 }")

    # Keep co-tenant + our total reservation <= CAP; reward is fixed, so the
    # remainder goes to rollout, clamped to [floor, 0.40].
    ROLLOUT_UTIL=$(awk "BEGIN{
        t = $CAP - $fmax - $REWARD_UTIL;
        if (t < $ROLLOUT_FLOOR) t = $ROLLOUT_FLOOR;
        if (t > 0.40) t = 0.40;
        printf \"%.2f\", t
    }")

    LOG=$REPO/logs/eval_step150_full_retry.attempt${attempt}.log

    {
        echo "=== attempt ${attempt}/${MAX_ATTEMPTS} | GPU ${GPUS} (neighbors ${f1},${f2}) | rollout=${ROLLOUT_UTIL} reward=${REWARD_UTIL} | $(date -u '+%F %T UTC') ==="
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
        echo "python: $(command -v python3) ($(python3 --version 2>&1))"
    } | tee -a "$MASTER_LOG"

    CUDA_VISIBLE_DEVICES="$GPUS" \
    bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_ocr.sh \
        trainer.val_only=True \
        trainer.experiment_name=wan22_5b_t2v_ocr_s150 \
        trainer.resume_mode=auto \
        data.val_files="$REPO/workspace/data/ocr/wan22/test.parquet" \
        trainer.validation_data_dir="$REPO/workspace/val_generations/eval_step150_full" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_UTIL" \
        reward.reward_model.rollout.gpu_memory_utilization="$REWARD_UTIL" \
        reward.reward_model.rollout.max_model_len=16384 \
        > "$LOG" 2>&1
    script_rc=$?

    if [ "$script_rc" -eq 0 ] && ! log_fatal "$LOG"; then
        echo "=== attempt ${attempt} COMPLETED at $(date -u '+%F %T UTC') ===" | tee -a "$MASTER_LOG"
        exit 0
    fi

    echo "=== attempt ${attempt} FAILED (rc=${script_rc}) at $(date -u '+%F %T UTC') ===" | tee -a "$MASTER_LOG"

    # Adapt: a KV-cache error means util was too low -> raise the rollout floor.
    if grep -q "No available memory for the cache blocks" "$LOG"; then
        ROLLOUT_FLOOR=$(awk "BEGIN{ t = $ROLLOUT_FLOOR + 0.04; if (t > 0.40) t = 0.40; printf \"%.2f\", t }")
        echo "    -> KV cache too small; raising rollout floor to ${ROLLOUT_FLOOR}" | tee -a "$MASTER_LOG"
    fi
    grep -c 'out of memory' "$LOG" 2>/dev/null | sed 's/^/    oom_lines=/' | tee -a "$MASTER_LOG"
    echo "=== sleeping ${RETRY_SLEEP}s before retry ===" | tee -a "$MASTER_LOG"
    sleep "$RETRY_SLEEP"
done

echo "=== gave up after ${MAX_ATTEMPTS} attempts at $(date -u '+%F %T UTC') ===" | tee -a "$MASTER_LOG"
exit 1
