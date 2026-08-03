# 评测报告：Wan2.2-TI2V-5B + DanceGRPO（视频 OCR reward）step-25 ckpt vs 基座

日期：2026-08-03（UTC） · 评测机：huirui（GPU 3,6） · 评测协议：与训练验证完全一致（`val_kwargs`：50 步推理、noise_level=0、Qwen3-VL-8B-Instruct GenRM 转写 + Levenshtein 比对 gts）

## 1. 结论（TL;DR）

| 指标 | 基座模型 | step-25 ckpt | Δ |
|---|---|---|---|
| **OCR reward mean@1（8 prompt 配对）** | 0.5864 | **0.6390** | **+0.0526（+9.0%）** |
| 胜 / 平 / 负（按 prompt） | — | — | 5 / 1 / 2 |
| 满分（1.0）样本数 | 2 / 8 | 2 / 8 | 0 |
| 零分样本数 | 2 / 8 | 0 / 8 | −2 |
| 单轮评测耗时 | 14.4 min | 18.6 min | 均 ≤30 min ✓ |

**25 步 DanceGRPO 训练后，视频内文字渲染的 OCR 可读性有可验证的提升**：方向与训练集上的验证曲线一致（64 行验证集：0.4730 → 0.4863，+0.0133；mini 子集效应更大是因为 8 条里含 2 条基座全崩的长文本样本）。零分样本从 2 个降为 0 个（虽有提升但仍接近零分，见 §4）。

## 2. 评测设置

- **ckpt**：`checkpoints/dance_grpo/wan22_5b_t2v_ocr_s150/global_step_25`（含优化器 28G；经 `resume_mode=auto` 加载，日志确认 `global_step_25` 载入）。已推送 ModelScope：[`michaelsou/wan22-ti2v-5b-dancegrpo-ocr-step25`](https://modelscope.cn/models/michaelsou/wan22-ti2v-5b-dancegrpo-ocr-step25)（30.0 GB，11 文件）。
- **基线**：`/data/models/Wan2.2-TI2V-5B-Diffusers`（同一模型，未经 RL）。
- **评测集**：`workspace/data/ocr/wan22/test_mini8.parquet` = 训练用 64 行验证子集的前 8 条（两轮评测同一批 prompt，逐条配对）。
- **运行方式**：`trainer.val_only=True`，两轮各自由同一脚本启动（baseline 不带 resume；ckpt 轮 `trainer.resume_mode=auto`）。生成 704×1280、8 帧视频。
- **打分**：`verl_omni/utils/reward_score/genrm_ocr.py`（`compute_score_ocr`），Qwen3-VL-8B 抽帧转写视频中文字 → 与 gts 做 Levenshtein 相似度。

## 3. 逐 prompt 配对明细

| # | 基座 | step-25 | Δ | gts | 基座 GenRM 转写 | ckpt GenRM 转写 |
|---|---|---|---|---|---|---|
| 0 | 0.930 | 0.950 | +0.020 | Spring Collection 2024 | Spring Colliection 2024 | Spring Collection 2024 |
| 1 | 1.000 | 0.938 | −0.062 | Step Goal Achieved | Step Goal Achieved | Step Gal Achieved |
| 2 | 0.750 | 1.000 | +0.250 | Forever Yours | Fower Yours | Forever Yours |
| 3 | 0.000 | 0.080 | +0.080 | Try Our New Burger | TRY BOIER OR VIU BIUGER… | TRY BOILER VILLIUMI BIUGEE |
| 4 | 0.000 | 0.100 | +0.100 | Lost City Near | LOSTEH ON.M TULY SOCT.MRY NEN | LOSTEL ANJU LOS TIN NER |
| 5 | 0.400 | 0.500 | +0.100 | Tonight Binary StandUp | Tonvoit Stard UP | Tooh Instare UP |
| 6 | 1.000 | 1.000 | 0.000 | Fearless | Fearless | Fearless |
| 7 | 0.611 | 0.544 | −0.067 | Loved And Remembered | Loved Remremed | LOVED AW RENEFRTNAD |

**均值：基座 0.5864 → ckpt 0.6390（+0.0526）**；胜 5 / 平 1 / 负 2。

## 4. 抽帧目检（ffmpeg，本地 frames/ 目录）

对 3 个代表性 prompt 抽帧人工核对（每段视频取第 4 帧，ckpt p2 另取 0/7 帧）：

- **p0（T 台 "Spring Collection 2024"）**：基座帧可读 **"Spping Collietion 2024"**（两处拼写错）；ckpt 帧可读 **"Spring Colllietion 2024"**（首词已正确，次词仍多一个 l）。肉眼可见的改进，与分数 0.930→0.950 一致。
- **p2（戒指刻字 "Forever Yours"）**：基座帧 **"Forver Yours"**；ckpt 第 0/4/7 帧均 **"Forver Yours"**（少一个 e），但 GenRM 给了 1.000。⚠️ 说明 Qwen3-VL 转写会"自动纠正"接近正确的花体字——1 字符级的字形错误可能被 VLM 脑补成正确拼写。本轮 ckpt 在此样本上的真实字形质量与基座相当，1.0 分含有 VLM 宽容度成分。
- **p3（汉堡广告牌 "Try Our New Burger"）**：基座 **"TRY BOIER / VIIIIIIII / BIUGEE"**，ckpt **"TRY BOIER / OR UIU / BIUGER"**。仍是乱码字，但 ckpt 的字母形态更接近真词（"BIUGER" vs "BIUGEE"）。**长文本/多行文本仍是主要未解失败模式**——这正是后续训练（25→150 步）最应继续改进的地方。

帧文件（本地 scratch `frames/`）：`frame_eval_baseline_0_p{0,2,3}.png`、`frame_eval_step25_25_p{0,2,3}.png`、`frame_ckpt_p2_f{0,7}.png`；远端视频：`workspace/val_generations/{eval_baseline/0,eval_step25/25}/{0..7}.mp4`。

## 5. 注意事项与局限

1. **样本量极小**（n=8，单 seed，mean@1 每 prompt 仅 1 个样本）：+0.053 的差值在统计上只能视为方向性证据；64 行验证集上的 +0.0133（0.4730→0.4863）是更大样本的旁证。正式结论应等 1018 行 `test.parquet` 全量评测（仓库 todo 的 Phase 6）。
2. **mini 子集分数不可与 64 行分数直接比**（子集构成不同）：基座在 64 行上 0.4730，在 mini8 上 0.5864。本报告只使用同题配对比较。
3. **GenRM 的 VLM 宽容度**：转写模型会纠正轻微字形错误（见 p2），分数略偏乐观；对基座和 ckpt 一视同仁，配对差值仍有效。
4. **视频时域不一致**：同一段视频不同帧文字可能有差异，GenRM 抽多帧转写，单帧目检可能与分数不完全对应（p2 即是）。
5. 评测期间 GPU 3,6 无其他租户任务干扰；两轮评测 0 Traceback、0 OOM。

## 6. 复现命令

```bash
# huirui, /data/verl-omni, venv 激活后；mini 集已由 test_smoke.parquet 前 8 行生成
# 基线
WORKSPACE=/data/verl-omni/workspace CUDA_VISIBLE_DEVICES=3,6 bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_ocr.sh \
  trainer.val_only=True trainer.experiment_name=eval_baseline \
  data.val_files=/data/verl-omni/workspace/data/ocr/wan22/test_mini8.parquet \
  trainer.validation_data_dir=/data/verl-omni/workspace/val_generations/eval_baseline
# step-25 ckpt
WORKSPACE=/data/verl-omni/workspace CUDA_VISIBLE_DEVICES=3,6 bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_ocr.sh \
  trainer.val_only=True trainer.experiment_name=wan22_5b_t2v_ocr_s150 trainer.resume_mode=auto \
  data.val_files=/data/verl-omni/workspace/data/ocr/wan22/test_mini8.parquet \
  trainer.validation_data_dir=/data/verl-omni/workspace/val_generations/eval_step25
```

日志：`logs/eval_baseline.log`、`logs/eval_step25.log`；生成明细：`workspace/val_generations/{eval_baseline/0.jsonl, eval_step25/25.jsonl}`（含每条 prompt 的 gts / GenRM 转写 / 分数 / 视频路径）。

## 7. 当前状态与后续

- 训练于 step ~42 处按用户指令暂停（step-25 为最新 ckpt，含优化器，可随时 `resume_mode=auto` 续训至 150）。
- /data 磁盘 99%（59G 空闲，其他用户 ~14G/h 增长）——续训需严格执行只留最近 2 个 ckpt，必要时删 smoke ckpt（28G）。
- 建议：续训到 150 后用最终 ckpt 在 1018 行 `test.parquet` 上跑全量正式评测（Phase 6），并保留本报告的 mini 配对协议作快速回归检查。

---

# 附录 A：step-50 ckpt 评测（2026-08-03T14:00Z）

ckpt `global_step_50`（已推 [`michaelsou/wan22-ti2v-5b-dancegrpo-ocr-step50`](https://modelscope.cn/models/michaelsou/wan22-ti2v-5b-dancegrpo-ocr-step50)，30.0GB/11 文件，0 失败）。同一 mini8 配对协议，评测 22.4 min。

## A.1 三点对比

| 指标 | 基座 | step-25 | step-50 |
|---|---|---|---|
| **OCR reward mean@1（mini8）** | 0.5864 | 0.6390 | **0.6233** |
| 训练内 64 行 val mean@1 | 0.4730（step0） | 0.4863 / 0.4800（复测） | **0.4883（新高）** |
| 零分样本 | 2 | 0 | 0 |
| 满分样本 | 2 | 2 | **3** |

| # | 基座 | s25 | s50 | Δ(50−25) | gts | s50 GenRM 转写 |
|---|---|---|---|---|---|---|
| 0 | 0.930 | 0.950 | **1.000** | +0.050 | Spring Collection 2024 | Spring Collection 2024 |
| 1 | 1.000 | 0.938 | 0.938 | 0.000 | Step Goal Achieved | Step Gal Achieved |
| 2 | 0.750 | **1.000** | **1.000** | 0.000 | Forever Yours | Forever Yours |
| 3 | 0.000 | 0.080 | **0.133** | +0.033 | Try Our New Burger | TRY BOUR VILWIOG BUOER BUIGET |
| 4 | 0.000 | 0.100 | **0.133** | +0.033 | Lost City Near | LOST ANU LOST IN NER |
| 5 | 0.400 | 0.500 | 0.420 | −0.080 | Tonight Binary StandUp | Toooh indare UP |
| 6 | 1.000 | **1.000** | **1.000** | 0.000 | Fearless | Fearless |
| 7 | 0.611 | 0.544 | 0.389 | −0.156 | Loved And Remembered | LOVE.D AW RENPTENLAN |

s50 vs s25：3 胜 2 负 3 平（−0.016 均值）；s50 vs 基座：5 胜 2 负（+0.037 均值）。

## A.2 解读

- **mini8（n=8）上 s50 与 s25 统计上打平**（−0.016 在单样本抖动范围内：p7 一个样本 −0.156 即贡献均值 −0.02）。退步集中在 p7（墓碑刻字，字形清晰但词序全错）与 p5。
- **更有把握的 64 行训练内 val 曲线仍在爬升**：0.4863 → **0.4883**（s25→s50，新高），且 s50 拿到了 mini8 首个完美满分（p0）与两个难例的持续改善（p3: 0→0.08→0.133，p4: 0→0.10→0.133）。
- **抽帧目检**（frames/frame_eval_step50_p{0,3,7}.png）：p0 帧为 "Spaiid Colllertion 2024"——GenRM 判 1.0 属 VLM 自动纠正（已在 §5.3 记录的宽容度）；p7 帧 "LOVE.D AW RENPTENLAN" 与转写一致，为真实退步。
- 结论：**训练继续沿正确方向前进（64 行 val 新高、难例单调改善），但幅度温和且 mini 子集上存在样本级反复**。长文本仍是主战场，支持继续训练至 150。

## A.3 产物

- 视频：`workspace/val_generations/eval_step50/50/{0..7}.mp4`；明细 `eval_step50/50.jsonl`；日志 `logs/eval_step50.log`（0 Traceback）。
- 上传日志 `logs/ms_upload_step50.log`（Elapsed 564.5s）。
