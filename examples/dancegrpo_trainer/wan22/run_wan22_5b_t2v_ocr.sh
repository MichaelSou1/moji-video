#!/bin/bash
# Wan2.2 RL with DanceGRPO on GPU, video OCR accuracy reward
#
# Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers (text+image-to-video, used in T2V mode)
# Algorithm: DanceGRPO (reuses FlowGRPO's advantage estimator and loss)
# Reward: GenRM OCR (Qwen3-VL-8B-Instruct transcription + Levenshtein vs ground truth)
#
# Derived from run_wan22_5b_t2v_hpsv3_npu.sh (NPU -> GPU port, HPSv3 -> OCR reward).
# Smoke-test scale: total_training_steps=20. Re-probe GPUs and override
# CUDA_VISIBLE_DEVICES / NUM_GPUS_ACTOR_ROLLOUT_REWARD before launching.

set -x

# huirui host driver is CUDA 12.8 while vllm 0.24 is a CUDA 13.3 build; the extracted
# cuda-compat user-mode driver bridges the gap. Only applied when the directory exists.
CUDA_COMPAT_DIR=/data/verl-omni/.cuda-compat/usr/local/cuda-13.3/compat
VENV_CU13_LIB=/data/verl-omni/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
if [ -d "$CUDA_COMPAT_DIR" ]; then
    export LD_LIBRARY_PATH=$CUDA_COMPAT_DIR:$VENV_CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/wan22/train.parquet
# Smoke runs validate on a 16-row subset (validation iterates the whole val set every
# pass; the full 1018-row test.parquet is for formal eval, Phase 6).
ocr_test_path=${OCR_VAL_PATH:-$WORKSPACE/data/ocr/wan22/test_smoke.parquet}

model_name=/data/models/Wan2.2-TI2V-5B-Diffusers
reward_model_name=/data/models/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

# GPU allocation: pick GPUs with the most free VRAM (nvidia-smi) before launching.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3,6}
export CUDA_VISIBLE_DEVICES
NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-2}
ROLLOUT_TP=1
REWARD_TP=1

ENGINE=vllm_omni
REWARD_ENGINE=vllm

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=dance_grpo \
    actor_rollout_ref.model.algorithm=dance_grpo \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.seed=42 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.attn_backend='native' \
    actor_rollout_ref.model.custom_chat_template='"{% if messages %}{% for message in messages %}{% if message[\"role\"] == \"user\" %}{{ message[\"content\"] }}{% endif %}{% endfor %}{% endif %}</s>"' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=10000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.height=704 \
    actor_rollout_ref.rollout.pipeline.width=1280 \
    actor_rollout_ref.rollout.pipeline.num_frames=8 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="dance_sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.max_model_len=32768 \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=dance_grpo \
    trainer.experiment_name=wan22_5b_t2v_ocr_s150 \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.validation_data_dir=$WORKSPACE/val_generations/wan22_5b_t2v_ocr_s150 \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=150 "$@" \
    2>&1 | tee run_wan22_5b_t2v_ocr.log
