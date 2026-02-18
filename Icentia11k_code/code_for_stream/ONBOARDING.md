# 从 0 快速上手：Icentia11k → LSL 流式 → WFDB 窗口

这份指南面向“第一次接触这套实现”的你：目标是用最短路径跑通、看懂输出、知道该从哪里改参数/接大模型，并能快速定位问题。

> 你现在的实现选择（已落地）：
> - **两路 LSL stream**：`ECG`（连续波形） + `ANN`（JSON markers）
> - **输出采样率**：默认 `250Hz`（保留 `--fs-out` 接口可调；支持真重采样）
> - **写盘窗口**：`window=10s, hop=10s`（不重叠）
> - **节律策略**：窗口开始注入“rhythm 自描述”（若窗口起点已在某节律内，则在 `.atr` 的 `sample=0` 写 `'(XXX'`）

---

## 0. 从哪里开始看（最推荐）

1) 先直接看“已经跑好的结果”（知道正确输出长什么样）：
- 最新报告入口：`ECG_Model/stream_result/report/README.md`
- 里面嵌了汇总图（10 秒窗口、背景节律、散点心拍标注）

2) 再看代码目录的总览（知道每个脚本职责）：
- `ECG_Model/code_for_stream/README.md`

---

## 1. 这套代码到底做了什么（用一句话）

把 Icentia11k 的 WFDB 长时数据（`.dat/.hea/.atr`）当作“真实设备采集的实时信号”，通过 LSL 以 **连续采样**的方式推送出去，同时把标注以 **事件流**推送出去；接收端把它们对齐后，按固定窗口写回 WFDB 小文件，便于你后续直接喂给大模型或其它算法模块。

---

## 2. 10 分钟复现：一条命令生成新结果（推荐）

如果你希望“完全从 0”自己跑一遍、生成一份新的 `stream_result`：

1) 安装依赖（建议用同一个解释器的 pip）：

```bash
python -m pip install -r ECG_Model/code_for_stream/requirements.txt
```

2) 一键跑完整链路（sender + receiver + 离线绘图 + 报告）：

```bash
python ECG_Model/code_for_stream/generate_stream_result.py --windows 3 --speed 20
```

它会在 `ECG_Model/stream_result/run_YYYYmmdd_HHMMSS/` 生成一份 run，并把“最新报告镜像”同步到 `ECG_Model/stream_result/report/`。

---

## 3. 30 分钟理解：拆开跑（知道每段在干嘛）

你未来要接大模型/改窗口/改 schema 时，建议你按下面顺序单独跑一次，理解“数据怎么流动”的：

### 3.1 Receiver：LSL → WFDB（写盘器）

Receiver 的职责是：订阅两路 LSL，把数据对齐后按窗口写 `.dat/.hea/.atr`。

```bash
python ECG_Model/code_for_stream/receiver_lsl_to_wfdb.py \
  --out-dir ECG_Model/stream_result/my_run/wfdb_250Hz \
  --window-sec 10 --hop-sec 10 \
  --inject-rhythm-state \
  --max-windows 5 \
  --verbose
```

### 3.2 （可选）Viewer：LSL 在线可视化

Viewer 用来快速确认“波形在动、标注在来、节律状态在变化”：

```bash
python ECG_Model/code_for_stream/viewer_lsl_realtime.py
```

如果你在无 GUI 环境（服务器/纯终端），用：

```bash
python ECG_Model/code_for_stream/viewer_lsl_realtime.py --no-plot --duration-sec 10
```

### 3.3 Sender：WFDB → LSL（流式模拟器）

Sender 的职责是：读 `.dat/.hea/.atr`，按 `--chunk-ms` 切块推送 ECG，并把 beat/rhythm 标注作为 JSON marker 推送到 ANN stream。

```bash
python ECG_Model/code_for_stream/sender_icentia_lsl.py \
  --dataset ECG_Model/dataset \
  --patient 0 \
  --fs-out 250 \
  --chunk-ms 200 \
  --speed 20 \
  --max-seconds 60 \
  --verbose
```

---

## 4. 输出怎么看（你后续要“理解结果/调参”的关键）

一次 run 的标准输出结构（以 `run_*/` 为例）：

- `wfdb_250Hz/`
  - `RECORDS`：窗口索引（每行是一个 record 名，如 `stream_000001`）
  - `stream_000000.dat/.hea/.atr`：一个 10 秒窗口
- `report/`
  - `assets/250Hz/summary.png`：多窗口汇总图
  - `assets/250Hz/stream_000000.png`：单窗口图
- `logs/`
  - sender/receiver/plot 的日志（窗口写不出来、速率异常时先看这里）

### 4.1 WFDB 窗口里的“对齐信息”在哪

Receiver 会把对齐用的元数据写进 `.hea` 的 `comments`，比如：
- `global_start_sample=...`：这个窗口对应的全局起点（在 `fs_out` 的 sample 域）
- `source_record_at_start=p00000_s00`：窗口起点属于哪个原始 segment

这对你未来做“跨窗口追踪/回溯源数据”非常重要。

---

## 5. 两路 LSL stream 的数据契约（你接大模型必须懂）

### 5.1 ECG stream（连续采样）

- 名字：默认 `icentia_ecg`
- 类型：`float32`
- 单位：`mV`
- 采样率：`fs_out`（默认 250Hz）

### 5.2 ANN stream（JSON marker 事件流）

- 名字：默认 `icentia_ann`
- 每条 marker 是一个 JSON 字符串（1 个 channel）
- 最重要字段：
  - `kind`：`annotation / segment_start / segment_end / session_start / session_end / ecg_chunk_end`
  - `global_sample`：**跨 segment 累计**的全局 sample index（基于 `fs_out`）
  - `symbol`：beat 时是 `N/S/V/Q`；节律 token 用 `+`
  - `aux_note`：只有 `symbol == '+'` 时常用，形如 `'(N' '(AFIB' '(AFL' ')'`

为什么要有 `global_sample`：
- 因为 ECG 是连续流、ANN 是事件流，二者必须靠一个共同坐标系对齐；
- `global_sample` 就是这个“共同坐标系”。

为什么要有 `ecg_chunk_end` watermark：
- 在高倍速下，ECG chunk 可能先到而 ANN marker 还在路上；
- Receiver 用 watermark 判断“到某个 sample 为止的 marker 都已经到齐了”，从而避免窗口写出时漏标注。

---

## 6. 可视化策略（如何评估“流式输出质量”）

你现在有两层可视化：

1) **在线**（调参最方便）：`viewer_lsl_realtime.py`
- 看 rolling ECG + beat 点 + 当前节律状态（背景色）
- 适合调 `--speed/--chunk-ms/--fs-out` 的时候快速反馈

2) **离线**（便于留档对比）：`plot_wfdb_windows.py`
- 读取 Receiver 写回的 WFDB 窗口，画成 PNG
- 把节律区域做背景色，把 beat 做散点
- 适合做“每次改动前后对比”

评估时建议你重点关注：
- beat 散点是否落在合理的心拍位置（通常是 QRS 附近）
- 节律背景色是否与 `.atr` 的 token 变化一致
- 窗口边界是否“自描述”（注入策略是否生效）

---

## 7. 你后续最常改的参数（快速记忆版）

### Sender（流式模拟）
- `--speed`：播放倍速（开发期强烈建议 10~50）
- `--fs-out`：输出采样率（真重采样，波形+标注都会映射）
- `--chunk-ms`：推流 chunk 大小（延迟 vs 开销）
- `--segment-start/--segment-end`：只播某些 segment（开发期非常有用）
- `--max-seconds`：只播前 N 秒（避免一次跑很久）

### Receiver（写盘窗口）
- `--window-sec`：窗口时长
- `--hop-sec`：步进（当前实现限定 `hop==window`，未来要做重叠滑窗可扩展）
- `--inject-rhythm-state`：窗口起点节律自描述注入（你已选择开启）

---

## 8. 代码地图（你要“快速读懂”的顺序）

建议阅读顺序（从“最像产品入口”到“最底层”）：

1) `generate_stream_result.py`：一键 demo / 报告生成
2) `sender_icentia_lsl.py`：WFDB → LSL（两路 stream、marker schema、水位线）
3) `receiver_lsl_to_wfdb.py`：LSL → WFDB（窗口化、watermark 对齐、rhythm 注入）
4) `plot_wfdb_windows.py`：离线评估可视化
5) `viewer_lsl_realtime.py`：在线评估可视化
6) `icentia_wfdb.py`：WFDB 读取、真重采样、label 规范化
7) `lsl_utils.py`：解决 `pylsl` 找不到 `liblsl` 的运行时问题（vendored so）

---

## 9. 常见坑（你遇到问题先看这里）

1) 你在 REPL/Notebook 里直接 `import pylsl` 报 “liblsl not found”
- 这是正常的：`pip install pylsl` 在一些环境只装了 Python binding
- 本项目脚本都用 `lsl_utils.import_pylsl()` 自动加载 vendored 的 `liblsl`（无需你手配）

2) 高倍速下窗口写出来但标注很少/没有
- 先确认 Sender 的 `--watermark-every-chunks` 没改大（默认 1）
- 先按建议顺序启动（Receiver/Viewer 先启动）
- 或让 Sender 保持默认 `--wait-for-consumers`（会等 consumer 连上再开始推流）

3) 想跑全量 50 段（很久）
- 先用 `--speed` + `--max-seconds` 迭代开发
- 真要全量再把限制去掉（否则一次迭代要等很久）

