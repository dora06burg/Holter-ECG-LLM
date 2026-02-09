# Icentia11k 流式模拟（LSL）报告（latest）

这是 `ECG_Model/stream_result/` 下的“最新报告镜像”，对应的 run 目录为：

- `ECG_Model/stream_result/run_20260209_163829/`

建议你直接从这里开始看：

- `ECG_Model/stream_result/report/assets/250Hz/summary.png`
- `ECG_Model/stream_result/report/README.md`（就是本文件）

---

## 1. 这次我帮你跑了什么（结果在哪里）

本次演示：把 Icentia11k 的 WFDB 数据（`.dat/.hea/.atr`）模拟成 **LSL 流式输入**，再把流式数据按 10 秒窗口写回 WFDB 三件套，并生成可视化图，方便你快速评估“流式输出效果”和“标注对齐是否正确”。

输出目录：

- WFDB 窗口：`ECG_Model/stream_result/run_20260209_163829/wfdb_250Hz/`
- 可视化：`ECG_Model/stream_result/report/assets/250Hz/`
- 日志：`ECG_Model/stream_result/run_20260209_163829/logs/`

---

## 2. 先看结果图：怎么解读（你最关心的“效果评估”）

![](assets/250Hz/summary.png)

每个子图对应一个 10 秒窗口：
- 黑色曲线：ECG（单位 mV）
- 彩色背景：节律（rhythm）
  - NSR：淡绿色
  - AFIB：淡红色
  - AFL：淡黄色
- 散点：心拍级标注（beat），点的位置在标注 sample 对应的波形上

你也可以看单窗口图：
- `ECG_Model/stream_result/report/assets/250Hz/stream_000000.png`
- `ECG_Model/stream_result/report/assets/250Hz/stream_000001.png`
- `ECG_Model/stream_result/report/assets/250Hz/stream_000002.png`

---

## 3. “节律自描述注入”是否生效？

你选择的策略是：**窗口开始如果已经处于某个节律，就在该窗口 `.atr` 的 sample=0 写一个 `'(XXX'`**，保证每个窗口都“自描述”。

这次运行可以在第二个窗口看到注入：
- `ECG_Model/stream_result/run_20260209_163829/wfdb_250Hz/stream_000001.atr` 的第 1 条标注是：`sample=0, symbol='+', aux_note='(N'`
- 这表示：窗口开始时处于 `N`（映射成 NSR）

快速验证命令：

```bash
python - <<'PY'
import wfdb
ann = wfdb.rdann("ECG_Model/stream_result/run_20260209_163829/wfdb_250Hz/stream_000001", "atr")
print(list(zip(ann.sample[:5], ann.symbol[:5], ann.aux_note[:5])))
PY
```

---

## 3.1 非正常心拍（Q）示例：segment 45（更直观）

主 run（`p00000_s00` 开头）前几十秒大多是 `N`，为了让你更直观地看到 **不同 beat 类型的散点叠加**，我额外录了一段只播放 `p00000_s45` 的短演示（开头包含很多 `Q`）。

对应的 WFDB 窗口在：
- `ECG_Model/stream_result/run_20260209_163829/wfdb_250Hz_qdemo_seg45/`

![](assets/qdemo_seg45/summary.png)

---

## 4. 快速上手：你自己怎么跑

依赖：

```bash
python -m pip install -r ECG_Model/code_for_stream/requirements.txt
```

建议启动顺序（避免高倍速丢开头）：

1) Receiver（写盘）：

```bash
python ECG_Model/code_for_stream/receiver_lsl_to_wfdb.py \
  --out-dir ECG_Model/stream_result/my_run/wfdb_250Hz \
  --window-sec 10 --hop-sec 10 \
  --inject-rhythm-state \
  --max-windows 10 \
  --verbose
```

2) Sender（推流）：

```bash
python ECG_Model/code_for_stream/sender_icentia_lsl.py \
  --dataset ECG_Model/dataset \
  --patient 0 \
  --fs-out 250 \
  --chunk-ms 200 \
  --speed 20 \
  --max-seconds 100 \
  --verbose
```

更详细的原理解释与参数说明在：

- `ECG_Model/stream_result/run_20260209_163829/report/README.md`
- `ECG_Model/code_for_stream/README.md`
