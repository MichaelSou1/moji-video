# Wan2.2 视频文字渲染强化学习（DanceGRPO + OCR Reward）

> 基于 [verl-omni](https://github.com/verl-project/verl-omni) 框架的实验项目：用 RL（DanceGRPO）后训练
> **Wan2.2-TI2V-5B** 视频生成模型，提升其生成视频中**文字渲染的准确性**（video text rendering accuracy）。
>
> 本仓库只做"接线"和少量扩展，不修改 verl-omni 的算法与 reward 实现。

最后更新：2026-08-02

## 目标

让 Wan2.2 生成的视频中的文字（招牌、标签、屏幕文字等）渲染得更准确。方法：

- **算法**：DanceGRPO（FlowGRPO 的 SDE 变体，verl-omni 原生支持，纯配置切换）
- **Reward**：`genrm_ocr.py::compute_score_ocr`——用 Qwen3-VL-8B 识别生成视频帧中的文字，
  与 ground truth 做 Levenshtein 相似度打分（原生支持 5D 视频张量，零代码改动）
- **数据**：`flow_grpo/dataset/ocr/`（19653 条训练 / 1018 条测试 prompt，目标文字以英文双引号标注，
  T2I prompt 直接用于 T2V）
- **硬件**：GPU（官方 DanceGRPO 脚本为 NPU 版，本项目已完成 GPU 移植）

## 当前状态：Phase 0–4 已完成 ✅

smoke test 已在 huirui GPU 服务器（2× NVIDIA H20 98GB）上端到端跑通：

| 验证项 | 结果 |
|---|---|
| 训练完成度 | 20/20 步，无崩溃、无 OOM |
| Reward 信号 | 每步均值 0.22–0.61，非 NaN、非全零；单样本 0.0–0.84，区分度良好 |
| 验证集 OCR 分数 | **0.4295（step 0，基座）→ 0.4683（step 20）**，20 步已见初步提升 |
| 验证视频语义 | 落盘视频中肉眼可见目标文字渲染尝试（如戒指刻字 "Fower Yours" → GT "Forever Yours"） |
| 资源消耗 | 峰值显存 ~61 GB/卡（余量 ~37 GB），约 19–27 分钟/步（32 视频/步） |
| Reward 单元验证 | 含目标文字输入 0.604 vs 无关文字输入 0.083（真实 Qwen3-VL-8B + 原版 `compute_score_ocr`） |

任务分解与验收标准见 [todo.md](todo.md)；Phase 5（正式训练）、Phase 6（前后对比评测）尚未开始。

## 快速开始

### 1. 环境（以 huirui 服务器为例）

```bash
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev,ocr]"
```

huirui 特有两坑（已在启动脚本中内置修复）：

- vllm 0.24 为 CUDA 13.3 构建而宿主驱动仅 CUDA 12.8 → 需 `cuda-compat` 用户态驱动
  （脚本中的 `LD_LIBRARY_PATH` 守卫段，免 root）
- PyPI 直连代理极慢 → 用镜像源，如 `export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`

### 2. 模型权重（/data/models/）

- `Wan-AI/Wan2.2-TI2V-5B-Diffusers`（~24 GB，基座）
- `Qwen/Qwen3-VL-8B-Instruct`（~8 GB，OCR reward VLM）

### 3. 数据处理

```bash
python3 examples/dancegrpo_trainer/data_process/wan22_ocr.py \
  --input_dir flow_grpo/dataset/ocr \
  --output_dir $WORKSPACE/data/ocr/wan22 \
  --frame_interval 2 --smoke_val_size 16
```

产出 `train.parquet` / `test.parquet` / `test_smoke.parquet`（小验证集，正式训练可用
`--smoke_val_size 64` 重建）。每行：`prompt`、`negative_prompt`、`ability="t2v"`、
`reward_model.ground_truth`（双引号内目标文字）、`extra_info.frame_interval`。

### 4. 启动训练

```bash
# 先探测 GPU 空闲显存（共享机器！），在脚本顶部设置 CUDA_VISIBLE_DEVICES
nvidia-smi --query-gpu=index,memory.free --format=csv

WORKSPACE=/data/verl-omni/workspace \
  bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_ocr.sh
```

脚本默认为 smoke 规模（`total_training_steps=20`，约 6–9 小时 @ 2×H20）。
正式训练建议：`total_training_steps=150`、`save_freq=test_freq=25`、
`experiment_name=wan22_5b_t2v_ocr_s150`、64 行验证子集（约 2.5–3 天 @ 2×H20），
支持 `trainer.resume_mode=auto` 断点续训。

## 本项目新增/修改的文件

| 文件 | 说明 |
|---|---|
| `examples/dancegrpo_trainer/data_process/wan22_ocr.py` | OCR 数据处理（双引号提取 ground truth，不过滤中文行，附带 smoke 验证子集） |
| `examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_ocr.sh` | GPU 版训练脚本（自官方 NPU 脚本最小化移植：去 Ascend 环境、`native`+`TORCH_SDPA` 注意力配对、GenRM OCR reward 接线、cuda-compat 修复） |
| `verl_omni/trainer/diffusion/ray_diffusion_trainer.py` | 修复 `_dump_generations` 不支持 Wan22 6D `[N,1,F,H,W,C]` 视频布局的问题（4 行 shape 归一化；非算法/reward 代码） |
| `tests/trainer/diffusion/test_dump_generations_video_on_cpu.py` | 上述修复的回归测试（5/5 通过） |

算法（`diffusion_algos.py`）与 reward（`genrm_ocr.py`）**零改动**。

## 关键架构决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 算法 | `dance_grpo` + `dance_sde`（window 2, range [0,5], noise 1.2） | 官方推荐用于 Wan2.2，数值更稳定，纯配置 |
| Reward | GenRM OCR（Qwen3-VL-8B + Levenshtein） | 已原生支持 5D 视频张量；VLM 对艺术字/透视文字比 PaddleOCR 鲁棒 |
| 分辨率/帧数 | 704×1280×8 帧（rollout），验证用 50 步确定性采样 | 官方 smoke 默认；实测文字可识别、reward 有区分度 |
| 2 卡并行 | `ROLLOUT_TP=1`、`REWARD_TP=1`（每卡完整副本，纯数据并行） | 实测 ~61 GB/卡，H20 98GB 余量充足 |
| `frame_interval` | 2（8 帧抽 4 帧评分） | 帧数少时多帧参与评分，信号更稳 |

## 训练监控要点

- `critic/rewards/mean`：期望噪声中缓慢上行（基线 val ≈ 0.43）
- `critic/rewards/zero_std_ratio`：持续过高说明 reward 饱和或任务过难/过易
- `val/reward/mean@1`：各验证点的 OCR 分数曲线（AC2 验收依据）
- 显存/磁盘：每卡应稳定 ~61 GB；`/data` 磁盘紧张，注意 ckpt 体积（首个 ckpt 落盘后立即检查）

## 路线图

- [x] Phase 0–4：环境、数据、reward 接线、GPU 移植、smoke test（本文"当前状态"）
- [ ] Phase 5：正式训练（150–300 步，监控 reward 上升趋势与 `zero_std_ratio`）
- [ ] Phase 6：训练前/后在完整 1017 条测试集上的 OCR 分数对比 + 抽样可视化
- [ ] Phase 7：收尾整理，可选回馈上游（须遵循 AGENTS.md 贡献规范）

## 致谢

本项目构建于 [verl](https://github.com/verl-project/verl) 与
[verl-omni](https://github.com/verl-project/verl-omni) 之上，rollout 推理使用
[vLLM-Omni](https://github.com/vllm-project/vllm-omni)；OCR 数据集来自
[flow_grpo](https://github.com/yifan123/flow_grpo)。算法与 reward 实现均为上游成果，
本项目仅做集成与工程移植。上游许可证：Apache 2.0（见 [LICENSE](LICENSE)）。
