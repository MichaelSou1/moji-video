# TODO: verl-omni + Wan2.2 + DanceGRPO + 视频 OCR Reward

> 目标：用 RL（DanceGRPO）后训练 `Wan2.2-TI2V-5B`，提升其生成视频中**文字渲染的准确性**（video text
> rendering accuracy），复用 verl-omni 已有的 DanceGRPO 算法实现和 OCR reward 实现，只做"接线"和少量
> 扩展，不做算法层面的重新开发。
>
> 本文档 = **Spec**（背景/目标/架构决策）+ **Steps**（分阶段可执行任务清单）+ **验收标准**。
> 执行时按 Phase 顺序推进；每个 Phase 内的 Step 可勾选跟踪。

---

## Part 1 — Spec

### 1.1 背景

- verl-omni 已经原生支持 **DanceGRPO**（FlowGRPO 的 SDE 变体，见
  `verl_omni/trainer/diffusion/diffusion_algos.py:269` 的
  `@register_diffusion_loss("dance_grpo")`，与 `flow_grpo` 复用同一个 loss 类），并提供了
  `Wan2.2-TI2V-5B` 的官方 DanceGRPO recipe：
  `examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_npu.sh`（**HPSv3 人类偏好 reward，非 OCR**）。
- verl-omni 已经实现了一个通用的 **视频/图像 OCR reward**：
  `verl_omni/utils/reward_score/genrm_ocr.py::compute_score_ocr`。它通过 VLM（Generative Reward
  Model，如 `Qwen/Qwen3-VL-8B-Instruct`）识别生成图像/视频中的文字，再用 Levenshtein 距离与
  ground-truth 文本比较打分，且**已经原生支持 5D video tensor `[B, F, H, W, C]`**（`genrm_ocr.py`
  第 122 行注释明确写了 "Wan22 DanceGRPO"），也就是说这条 reward 路径本来就是为这个场景准备的。
- 原始 `flow_grpo` 仓库（已放在本仓库 `flow_grpo/` 下作为参考）中有一条 **Wan2.1 + video OCR**
  的先例：`flow_grpo/config/grpo.py::general_ocr_wan2_1`（标准 FlowGRPO，非 DanceGRPO）+
  `flow_grpo/flow_grpo/ocr.py::OcrScorer_video_or_image`（用 **PaddleOCR** 而非 VLM 做识别）。
  这条路径不直接可用（是旧框架 + 旧算法），但其数据/超参选择（分辨率 240×416、33 帧、
  `frame_interval` 抽帧评估）可作为参数参考。
- OCR 训练数据来自 `flow_grpo/dataset/ocr/{train,test}.txt`（本仓库已有，19652/1017 行），每行一条
  prompt，要渲染的文字用英文双引号包裹，例如：
  `A close-up of a medicine bottle ... that reads "Take With Food" ...`。这批 prompt 本身是为
  T2I 设计的，但完全可以直接喂给 T2V 模型（把"生成图片"换成"生成视频"，文字仍然是需要出现在画面里
  的目标文本），`examples/flowgrpo_trainer/data_process/qwenimage_ocr.py` 和
  `examples/dancegrpo_trainer/data_process/wan22_hpsv3.py` 是两条现成的数据处理管线模板。

### 1.2 目标（Definition of Done）

1. 能用一条命令启动 `Wan2.2-TI2V-5B` 的 DanceGRPO 训练，reward 为视频 OCR 准确率。
2. 训练过程中 reward/OCR-score 曲线随 step 上升，且验证集上抽取的视频经人工/工具复核文字渲染
   确实变准。
3. 产出一份可复现的评测结果：**训练前 vs 训练后** 在固定测试集上的视频 OCR 准确率对比（数值 + 抽样
   可视化）。
4. 全部改动可运行在 GPU 环境（现有官方脚本是 NPU 版，需要移植），并已在 huirui 服务器
   （`/data/verl-omni`，见仓库根目录 `MEMORY.md`）跑通至少一个短程 smoke test。

### 1.3 关键架构决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 算法 | `dance_grpo`（复用 FlowGRPO loss，SDE 用 `dance_sde`） | 官方推荐用于 Wan2.2，数值更稳定；无需改代码，纯配置切换 |
| Reward 实现 | 优先用现成的 `genrm_ocr.py::compute_score_ocr`（VLM 识别 + Levenshtein） | 已原生支持 5D video tensor，零代码改动；VLM 对艺术字/透视文字比 PaddleOCR 更鲁棒 |
| Reward 备选 | 如 VLM reward 效果不理想/资源不够，退化为移植 `flow_grpo/flow_grpo/ocr.py::OcrScorer_video_or_image`（PaddleOCR）为新的 `compute_score_paddle_ocr` | PaddleOCR 更轻量、无需额外部署 VLM 推理服务，但对复杂字体/视角鲁棒性较差 |
| 数据来源 | 复用 `flow_grpo/dataset/ocr/{train,test}.txt`（T2I OCR prompt 直接用于 T2V） | 现成、体量够大（19652/1017条），无需自建数据集；后续可选补充真正为视频设计的文字提示词 |
| 硬件 | GPU（现有 DanceGRPO 脚本是 NPU 版，需要移植参数） | 用户环境是 huirui GPU 服务器，非昇腾 NPU |
| 视频分辨率/帧数 | 先用官方 HPSv3 脚本的 704×1280×8帧（rollout）做 smoke test，正式训练视需要文字清晰度上调分辨率或参考 `flow_grpo` 的 Wan2.1 配置 (240×416×33帧) 做取舍 | 分辨率越高文字越清晰但显存/时间成本越高，需要实验验证的超参 |

### 1.4 风险与已知坑

- **视频分辨率 vs 文字可读性**：DanceGRPO 官方 HPSv3 脚本用的分辨率较低、帧数只有 8 帧（`height=704
  width=1280 num_frames=8`），这对人类偏好评分够用，但对 OCR 而言文字可能因分辨率/帧数不足而无法被
  VLM/PaddleOCR 稳定识别到。**需要在小规模试验中先验证 reward 信号是否有效（reward 方差不为 0），
  再决定是否上调分辨率**。
- **GPU 显存**：`Wan-AI/Wan2.2-TI2V-5B-Diffusers` 是 50 亿参数的视频扩散模型，叠加 DanceGRPO 的
  `rollout.n`（group size，官方设为 8）会显著放大显存需求。需要先确认可用 GPU 数量与显存，参考
  `docs/start/flowgrpo_quickstart.md` 的 OOM 调参表分配 `ROLLOUT_TP`/`REWARD_TP`。
- **奖励稀疏/文字太小**：视频中文字如果只占画面很小区域，reward 信号会很稀疏，训练可能不收敛。
  建议 prompt 筛选或改写时优先选择"大字号/特写"类描述（`flow_grpo` 的 OCR prompt 本身已偏向
  "close-up"、"clear label"等描述，基本满足要求）。
- **VLM reward 的推理成本**：`genrm_ocr.py` 每个 frame 都会调用一次 VLM，视频抽帧数越多，reward
  计算延迟越高，需要合理设置 `extra_info["frame_interval"]`（4/8 帧抽一帧）或采用 async reward
  资源池方案（参考 `run_qwen_image_ocr_lora_async_reward.sh`）。
- **中文文字**：`wan22_hpsv3.py` 的数据处理脚本会**过滤掉含中文字符的行**（`_contains_chinese`）。
  `flow_grpo/dataset/ocr` 数据集本身也是纯英文文字渲染任务。如果目标是中文文字渲染准确性，需要
  另外准备中文 OCR 提示词数据集（当前不在默认范围内，见 Phase 5 可选任务）。

---

## Part 2 — Steps

### Phase 0：环境与基线确认

- [ ] 0.1 确认 huirui 服务器 GPU 型号、数量、单卡显存（`nvidia-smi`），据此决定 Phase 3 的
      `NUM_GPUS`/`ROLLOUT_TP`/`REWARD_TP` 取值。
- [ ] 0.2 下载/确认模型权重（huirui 上当前模型资产见下表，需从 ModelScope 补齐缺失项）。

  **huirui 上已有模型资产（核实于 2025-08）**

  | 模型 | ModelScope ID | 位置 | 状态 | 体积 |
  |------|--------------|------|------|------|
  | **Wan2.2-TI2V-5B-Diffusers** (base model) | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `~/.cache/huggingface/hub/` ref only | ❌ **需下载** | ~24 GB |
  | Wan2.1-T2V-1.3B (参考，非必需) | `Wan-AI/Wan2.1-T2V-1.3B` | `~/VEGA-3D/data/models/` | ✅ 已有 | 5.8 GB |
  | **Qwen3-VL-8B-Instruct** (OCR VLM，推荐) | `Qwen/Qwen3-VL-8B-Instruct` | `~/.cache/huggingface/hub/` 部分（仅 395M blobs，缺 safetensors） | ⚠️ **需补全** | ~8 GB |
  | Qwen3-VL-30B-A3B-Instruct (过大，非必需) | — | `/data/models/` | ✅ 完整 | 58 GB |
  | Qwen2.5-VL-7B-Instruct (备选 VLM) | `Qwen/Qwen2.5-VL-7B-Instruct` | `~/.cache/huggingface/hub/` ref only | ❌ 需下载 | ~15 GB |

  > ⚠️ `/data` 磁盘已用 97%（3.2T/3.5T，仅剩 ~116G），多人共享。下载前确认空间足够，优先下载
  > 必需的两个模型（Wan2.2 ~24G + Qwen3-VL-8B ~8G ≈ 32G），不建议下载 Qwen3-VL-30B（58G）。

  下载命令（在 huirui 上执行，**先 `bash -ic "clashon"` 开代理**）：
  ```bash
  # 安装 modelscope CLI（如未装）
  pip install modelscope

  # ① Base model: Wan2.2-TI2V-5B-Diffusers（~24 GB，约 30-60 min）
  modelscope download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local_dir /data/models/Wan2.2-TI2V-5B-Diffusers

  # ② OCR VLM: Qwen3-VL-8B-Instruct（~8 GB，约 10-20 min）
  modelscope download Qwen/Qwen3-VL-8B-Instruct \
    --local_dir /data/models/Qwen3-VL-8B-Instruct
  ```

  下载完成后验证：
  ```bash
  # Wan2.2 应含 diffusion_pytorch_model-*.safetensors (3个分片) + config.json + VAE 权重
  ls /data/models/Wan2.2-TI2V-5B-Diffusers/*.safetensors | wc -l   # 预期 ≥3
  # Qwen3-VL 应含 model-*.safetensors 分片 + tokenizer 文件
  ls /data/models/Qwen3-VL-8B-Instruct/*.safetensors | wc -l      # 预期 ≥1
  ```

  后续训练脚本中通过 `model_name=/data/models/Wan2.2-TI2V-5B-Diffusers` 和
  `reward_model_name=/data/models/Qwen3-VL-8B-Instruct` 引用本地路径，避免运行时再从 Hub 拉取。

- [ ] 0.3 按 `docs/start/install.md` 在 huirui 上完成 GPU 环境安装：
  ```bash
  uv venv --python 3.12 --seed && source .venv/bin/activate
  uv pip install -e ".[gpu]" --torch-backend=auto
  uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
  uv pip install -e ".[train,dev,ocr]"   # ocr extra 提供 Levenshtein
  ```
- [ ] 0.4 跑一遍 `docs/start/install.md` 的 Post-Installation Verification 五条 `python -c` 检查。
- [ ] 0.5 （可选但强烈建议）先按官方 quickstart 跑通一次现成的 FlowGRPO/DanceGRPO 示例
      （如 `bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_npu.sh` 改成 GPU 版
      或直接跑 `run_qwen_image_ocr_lora.sh`），验证 Ray/vLLM-Omni 基础设施可用，排除环境问题
      对后续调试的干扰。

### Phase 1：数据集准备

- [ ] 1.1 确认数据来源：直接复用仓库自带的 `flow_grpo/dataset/ocr/{train,test}.txt`（无需下载，
      已在本仓库中）。如需扩充数据量或改善 prompt 质量，可从
      [flow_grpo GitHub](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr) 拉取最新版本。
- [ ] 1.2 新建视频 OCR 专用数据处理脚本
      `examples/dancegrpo_trainer/data_process/wan22_ocr.py`，参考两个现有模板融合而成：
      - 参考 `examples/flowgrpo_trainer/data_process/qwenimage_ocr.py` 的 `extract_solution`
        （从 `"..."` 中提取 ground-truth 文字）和 `reward_model.ground_truth` 字段写法；
      - 参考 `examples/dancegrpo_trainer/data_process/wan22_hpsv3.py` 的 `ability="t2v"`、
        `negative_prompt` 结构、`data_source` 命名习惯（建议 `data_source = "dance_grpo/ocr"`）。
      - 输出 schema（每行）：
        ```python
        {
            "data_source": "dance_grpo/ocr",
            "prompt": [{"role": "user", "content": <原始prompt文本>}],
            "negative_prompt": [{"role": "user", "content": " "}],
            "ability": "t2v",
            "reward_model": {"style": "model", "ground_truth": <提取出的目标文字>},
            "extra_info": {"split": <train/test>, "index": <idx>, "frame_interval": 4},
        }
        ```
      - 与 `wan22_hpsv3.py` 不同：**不要过滤中文行**除非确认目标只做英文文字（见 Spec 1.4）；
        且需要保留 `extract_solution` 的双引号提取逻辑（`wan22_hpsv3.py` 没有这个逻辑，因为 HPSv3
        不需要 ground-truth 文本，而 OCR 任务必须要）。
- [ ] 1.3 运行数据处理脚本，生成 parquet：
  ```bash
  python3 examples/dancegrpo_trainer/data_process/wan22_ocr.py \
    --input_dir flow_grpo/dataset/ocr \
    --output_dir $WORKSPACE/data/ocr/wan22
  ```
  产出 `$WORKSPACE/data/ocr/wan22/{train,test}.parquet`。
- [ ] 1.4 抽查 parquet 内容（`pandas.read_parquet` 打印几行），确认 `ground_truth` 字段被正确提取、
      无空字符串、无异常长度。

### Phase 2：Reward 函数接线（默认方案：VLM/GenRM OCR）

- [ ] 2.1 确认奖励模型：优先使用 Phase 0.2 已下载到 `/data/models/Qwen3-VL-8B-Instruct` 的
      `Qwen/Qwen3-VL-8B-Instruct`（与 Qwen-Image OCR 示例一致），或更小的备选
      `Qwen/Qwen2.5-VL-3B-Instruct`（显存需求更低，适合先做小规模验证，需额外下载约 6G）。
- [ ] 2.2 确认 `verl_omni/utils/reward_score/genrm_ocr.py::compute_score_ocr` **无需修改**即可用于
      Wan2.2 视频输出——已支持 `[B, F, H, W, C]` 5D tensor（对应注释里的 "Wan22 DanceGRPO"）。
      仅需在启动脚本里配置：
      - `reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py`
      - `reward.custom_reward_function.name=compute_score_ocr`
      - `reward.reward_model.enable=True`
      - `reward.reward_model.model_path=<VLM 路径>`
- [ ] 2.3 在小 batch 上做单元级验证：可写一个临时脚本直接调用 `compute_score_ocr`（用一段随机
      tensor 或已有 Wan2.2 生成的视频 + 已知 ground_truth），确认返回的 `score` 字段在
      `[0, 1]` 且不同输入分数有区分度（不是恒为 0 或恒为 1，否则训练无信号）。跑完后删除该临时脚本。
- [ ] 2.4 决定 `extra_info["frame_interval"]`：视频帧数较少时（如官方 HPSv3 脚本的 8 帧）设为 1
      或 2（尽量多帧参与评分，提高 OCR 信号稳定性）；帧数较多时可设为 4/8 降低 VLM 调用次数。

### Phase 2b（可选/备选）：Reward 函数改造为 PaddleOCR 方案

> 仅当 Phase 2 的 VLM reward 效果不佳、或希望规避额外部署 VLM 推理服务时执行。

- [ ] 2b.1 在 huirui 上安装 PaddleOCR 依赖：
  ```bash
  pip install paddlepaddle-gpu==2.6.2 paddleocr==2.9.1 python-Levenshtein
  ```
- [ ] 2b.2 新建 `verl_omni/utils/reward_score/paddle_ocr.py`，移植
      `flow_grpo/flow_grpo/ocr.py::OcrScorer_video_or_image` 的核心逻辑，但改写为符合 verl-omni
      reward 接口的函数签名（参考 `genrm_ocr.py` 的 `compute_score_ocr` 签名与 5D tensor 归一化
      逻辑，复用其 `_to_pil`/frame 提取代码模式）：
      ```python
      def compute_score_paddle_ocr(data_source, solution_image, ground_truth, extra_info, **kwargs) -> dict:
          ...
      ```
- [ ] 2b.3 在启动脚本中把
      `reward.custom_reward_function.name` 改成 `compute_score_paddle_ocr`，并**移除**
      `reward.reward_model.*` 相关配置（PaddleOCR 是规则式打分，不需要独立的 VLM 推理服务，
      参考 `docs/algo/flowgrpo.md` "Rule-Based Reward Training" 一节的做法）。
- [ ] 2b.4 同 Phase 2.3，做单元级验证。

### Phase 3：训练脚本改造（NPU → GPU，Wan2.2 + DanceGRPO + OCR）

- [ ] 3.1 复制官方 NPU 脚本作为起点：
  ```bash
  cp examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_npu.sh \
     examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_ocr.sh
  ```
- [ ] 3.2 移除/替换 NPU 专属内容（对照
      `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh` 的 GPU 写法）：
      - 删除顶部 `ASCEND_HOME_PATH`、`source .../set_env.sh`、
        `export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`。
      - 删除/改掉 `trainer.device=npu`（GPU 版留空即默认 CUDA）。
      - 将 `actor_rollout_ref.model.attn_backend='_native_npu'` 改为 GPU 对应的
        `_flash_3_varlen_hub`（并确认 `docs/start/install.md` 的 FA3 依赖已装，否则会自动 fallback
        到 native/SDPA，不影响正确性只影响速度）。
- [ ] 3.3 替换数据路径为 Phase 1 产出的 parquet：
  ```bash
  ocr_train_path=$WORKSPACE/data/ocr/wan22/train.parquet
  ocr_test_path=$WORKSPACE/data/ocr/wan22/test.parquet
  ```
- [ ] 3.4 替换 reward 配置块（删除 HPSv3 相关的 `custom_reward_model_path`、
      `hpsv3_reward.py`、`compute_score_hpsv3`），换成 Phase 2 确定的 OCR reward 配置：
  ```bash
  reward_model_name=/data/models/Qwen3-VL-8B-Instruct   # Phase 0.2 下载的本地路径
  reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py
  ...
  reward.reward_model.enable=True \
  reward.reward_model.model_path=$reward_model_name \
  reward.reward_model.rollout.name=vllm \
  reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
  reward.custom_reward_function.path=$reward_function_path \
  reward.custom_reward_function.name=compute_score_ocr \
  ```
- [ ] 3.5 根据 Phase 0.1 确认的 GPU 数量，重新分配 `NUM_GPUS_ACTOR_ROLLOUT_REWARD`、
      `ROLLOUT_TP`、`REWARD_TP`（参考 `docs/start/flowgrpo_quickstart.md` 的 OOM 调参表；显存不够
      优先增大 `ROLLOUT_TP`/`REWARD_TP` 或改用 async reward 资源池模式，参考
      `run_qwen_image_ocr_lora_async_reward.sh` 的 `reward.reward_model.enable_resource_pool=True`
      写法）。
- [ ] 3.6 保留/沿用 DanceGRPO 算法相关配置（这些不需要改）：
      `algorithm.adv_estimator=dance_grpo`、`actor_rollout_ref.model.algorithm=dance_grpo`、
      `actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo`、
      `actor_rollout_ref.rollout.algo.sde_type=dance_sde`、`sde_window_size=2`、
      `sde_window_range=[0,5]`、`noise_level=1.2`。
- [ ] 3.7 视频分辨率/帧数先按官方 HPSv3 默认值起步做 smoke test
      （`height=704 width=1280 num_frames=8 num_inference_steps=10`），验证跑通后再按 Spec 1.4
      的风险提示评估是否需要上调分辨率/帧数以提升文字清晰度。
- [ ] 3.8 调整 `trainer.*`：
      `trainer.project_name=dance_grpo`、`trainer.experiment_name=wan22_5b_t2v_ocr`、
      `trainer.logger`（wandb 需要 `export WANDB_API_KEY=...`，否则用
      `'["console", "tensorboard"]'`）、`trainer.log_val_generations`（建议 ≥4，便于人工抽查视频）、
      `trainer.val_before_train=True`（先看 base model 的初始 OCR 分数作为基线）。
- [ ] 3.9 `total_training_steps`/`total_epochs` 先设置成小值（如 `total_training_steps=20`）
      用于 Phase 4 的 smoke test，正式训练时再调大。

### Phase 4：Smoke Test（小规模跑通）

- [ ] 4.1 确认 Phase 0.2 下载的模型权重已在 `/data/models/` 下就绪（若尚未下载，先按 Phase 0.2
      的命令补齐，下载前 `bash -ic "clashon"` 开代理）。
- [ ] 4.2 用最小 GPU 数/最小 batch 跑几个 step，确认：
      - 训练进程能正常启动，不报 config/shape 错误；
      - reward/OCR-score 日志字段能正常打印（非 NaN、非全 0）；
      - 显存不 OOM（若 OOM，参考 `docs/start/flowgrpo_quickstart.md` 的调参表逐项排查）。
- [ ] 4.3 检查 `trainer.log_val_generations` 落盘的验证视频，人工确认视频里确实渲染了目标文字
      （哪怕不准确），验证整条 pipeline（prompt → rollout 生成视频 → OCR reward）语义正确。
- [ ] 4.4 记录 smoke test 阶段的每 step 耗时、显存占用，估算正式训练全量跑完的时间/资源开销。

### Phase 5：正式训练

- [ ] 5.1 根据 Phase 4 的资源评估，把 `total_epochs`/`total_training_steps` 调整为正式值
      （参考同类 OCR recipe 常见设置：`total_epochs=15`、`total_training_steps=300`，具体按
      收敛情况调整）。
- [ ] 5.2 设置 `trainer.save_freq`/`trainer.test_freq`（如每 30 step 存一次 ckpt + 跑一次验证）。
- [ ] 5.3 启动正式训练，持续监控（wandb/tensorboard）：
      - `reward`/OCR score 均值是否随 step 上升；
      - `zero_std_ratio`、`std_mean`（见 `docs/start/metrics.md`）：过高的 `zero_std_ratio`
        说明 reward 饱和或任务过难/过易，需要调整 prompt 难度分布或 reward 尺度；
      - `ratio_mean`/`ratio_std`、`pg_clipfrac_*`：判断是否需要调 `clip_range`/学习率。
- [ ] 5.4 若训练不稳定（reward 不涨、loss 发散），按 `docs/algo/flowgrpo.md` 的调参建议排查：
      先设 `beta=0`/`use_kl_loss=False` 看 reward 能否上升，再逐步引入正则化。
- [ ] 5.5 （可选）如显存/时间允许，做一次 `sde_type` 消融（`dance_sde` vs `sde` vs `cps`），
      验证官方推荐的 `dance_sde` 确实优于其他变体。

### Phase 6：评测

- [ ] 6.1 明确评测集：使用 Phase 1 产出的 `test.parquet`（1017 条，与训练集不重叠）。
- [ ] 6.2 训练前基线：用 base `Wan-AI/Wan2.2-TI2V-5B-Diffusers`（未经 RL 微调）在测试集上跑一遍
      推理 + OCR 打分，记录平均 OCR score（即 Phase 3.8 中 `trainer.val_before_train=True` 产出的
      第 0 步验证结果，或额外单独跑一次纯推理 + reward 脚本）。
- [ ] 6.3 训练后评测：用训练产出的 LoRA/全量权重 checkpoint，在同一测试集上重新推理 + OCR 打分。
      建议提高验证时的 `num_inference_steps`（如官方设置 `val_kwargs.pipeline.num_inference_steps=50`，
      `val_kwargs.algo.noise_level=0.0`，即验证时用确定性采样/更多步数，避免训练时的加速采样影响
      评测公平性）。
- [ ] 6.4 汇总对比表：训练前 vs 训练后的平均 OCR score（Levenshtein 归一化分数）、以及可选的
      "完全匹配率"（`ground_truth` 完整出现在识别文本中的比例，可从 `_levenshtein_score` 逻辑里
      的 `dist == 0` 条件衍生一个二值指标）。
- [ ] 6.5 抽样可视化：从测试集中随机挑 8-16 条 prompt，导出训练前/后生成的视频帧对比图（或视频
      本身），人工检查文字渲染是否确实更清晰/更准确，作为定量指标之外的定性佐证。
- [ ] 6.6 （可选）跨 reward 一致性检查：如果 Phase 2b 也实现了 PaddleOCR reward，可用它作为**独立于
      训练所用 reward 之外的第三方评测器**，避免"reward hacking"（模型只是学会了迎合训练用的那个
      VLM 的识别偏好，而非真的文字更准）——即用 VLM reward 训练，但用 PaddleOCR（或反之）做最终评测。

### Phase 7：收尾

- [ ] 7.1 清理 Phase 2.3/2b.4 中用于调试的临时脚本（不应遗留在仓库中）。
- [ ] 7.2 将本地新增文件（数据处理脚本、训练脚本、可能新增的 `paddle_ocr.py`）同步到 huirui
      （`rsync -avz -e ssh ./ huirui:/data/verl-omni/`，见仓库 `MEMORY.md`）。
- [ ] 7.3 若决定贡献回上游 `verl-project/verl-omni`（例如把
      `wan22_ocr.py`/`run_wan22_5b_t2v_ocr.sh` 作为新的官方 example），**必须先阅读并遵循**
      `.claude/skills/commit-and-pr/SKILL.md`（或 `AGENTS.md` 第 1 节）里的查重、PR 标题格式、
      commit trailer、AI 协助披露等强制要求，不要跳过。
- [ ] 7.4 整理最终结果（Phase 6 的评测表 + 可视化）沉淀为实验记录，方便后续复现或进一步迭代
      （例如换更大分辨率、换中文数据集）。

---

## Part 3 — 验收标准（Acceptance Criteria）

- [ ] AC1：`bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_ocr.sh` 能在 huirui GPU 服务器
      上一键跑通至少 20 个训练 step，无崩溃。
- [ ] AC2：训练日志中 reward（OCR score）均值相较第 0 step 有可观测的上升趋势。
- [ ] AC3：产出训练前/后在同一测试集（1017 条 prompt）上的 OCR score 对比数值，训练后显著高于
      训练前（具体阈值视实验结果而定，无需预设硬指标，但需有统计意义上的提升，而非噪声波动）。
- [ ] AC4：至少 8 组抽样视频的训练前/后对比可视化，人工确认文字渲染质量提升趋势与量化指标一致。
- [ ] AC5：全部新增脚本/代码已同步到 huirui 服务器仓库，且本地/远程 git 状态一致。

---

## 附：关键文件速查表

| 用途 | 路径 |
|---|---|
| DanceGRPO loss 实现（无需改） | `verl_omni/trainer/diffusion/diffusion_algos.py` |
| 官方 Wan2.2 DanceGRPO(HPSv3) 脚本模板 | `examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_npu.sh` |
| 官方 Wan2.2 HPSv3 数据处理模板 | `examples/dancegrpo_trainer/data_process/wan22_hpsv3.py` |
| 现成视频/图像 OCR reward（无需改） | `verl_omni/utils/reward_score/genrm_ocr.py` |
| GPU 版 OCR 训练脚本参考（Qwen-Image） | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh` |
| Async reward 资源池写法参考 | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_async_reward.sh` |
| Qwen-Image OCR 数据处理模板（含 ground_truth 提取） | `examples/flowgrpo_trainer/data_process/qwenimage_ocr.py` |
| 原始 OCR prompt 数据集 | `flow_grpo/dataset/ocr/{train,test}.txt` |
| 参考：PaddleOCR 视频打分实现（旧框架） | `flow_grpo/flow_grpo/ocr.py::OcrScorer_video_or_image` |
| 参考：Wan2.1 + OCR 超参（旧框架，标准FlowGRPO） | `flow_grpo/config/grpo.py::general_ocr_wan2_1` |
| 算法文档 | `docs/algo/flowgrpo.md`、`examples/dancegrpo_trainer/README.md` |
| 批大小/OOM 调参文档 | `docs/start/flowgrpo_quickstart.md`、`docs/algo/flowgrpo.md#batch-size` |
| 训练指标含义 | `docs/start/metrics.md` |
| huirui 模型权重存放位置 | `/data/models/`（Phase 0.2 下载） |
| 本地/远程环境与同步说明 | `MEMORY.md` |
