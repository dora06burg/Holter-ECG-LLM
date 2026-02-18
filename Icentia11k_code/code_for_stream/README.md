# Icentia11k 流式模拟（LSL）

本目录实现把 `ECG_Model/dataset/Icentia11k`（WFDB: `.dat/.hea/.atr`）模拟成真实检测场景的“流式输入”：

- **LSL stream 1：ECG**（连续 `float32`，单位 mV，默认 `250Hz`，可真重采样）
- **LSL stream 2：ANN**（事件/marker 流：每条为 JSON 字符串，包含 `global_sample/symbol/aux_note/...`）

同时提供：

- **Receiver**：把 LSL 流按窗口写回 WFDB 三件套（`.dat/.hea/.atr`）到输出文件夹  
  当前按你的选择实现：`window=10s, hop=10s`，并支持 **rhythm 自描述注入**（window 起点注入 `'(XXX'`）。
- **Viewer**：实时滚动可视化，方便评估流式效果与标注对齐。

从 0 快速上手（推荐先读）：

- `ECG_Model/code_for_stream/ONBOARDING.md`

## 依赖安装

Python 依赖：

```bash
python -m pip install -r ECG_Model/code_for_stream/requirements.txt
```

LSL 运行时库：

- `pip install pylsl` 只装 Python 绑定，有些 Linux 环境缺 `liblsl` 动态库。
- 本项目已在 `ECG_Model/code_for_stream/vendor/lsl_lib/` **vendored** 了可用的 `liblsl` 和 `libpugixml`；
  `lsl_utils.import_pylsl()` 会自动加载它们（无需手动设置 `LD_LIBRARY_PATH`）。

## 快速运行（建议顺序）

1) 先启动 Viewer（可选，但推荐用于快速确认）

```bash
python ECG_Model/code_for_stream/viewer_lsl_realtime.py
```

2) 启动 Receiver（把流写回 WFDB 小窗口文件夹）

```bash
python ECG_Model/code_for_stream/receiver_lsl_to_wfdb.py \\
  --out-dir ECG_Model/code_for_stream/output_wfdb \\
  --inject-rhythm-state \\
  --max-windows 5 \\
  --verbose
```

3) 启动 Sender（开始播放数据到 LSL）

```bash
python ECG_Model/code_for_stream/sender_icentia_lsl.py \\
  --dataset ECG_Model/dataset \\
  --patient 0 \\
  --fs-out 250 \\
  --chunk-ms 200 \\
  --speed 10 \\
  --max-seconds 60 \\
  --verbose
```

说明：
- `--speed 10` 表示 **10 倍速**，否则 50×70 分钟全量会非常久。
- `--fs-out` 支持与原始采样率不同，会做**真重采样**（波形 + 标注 sample index）。
- Sender 默认会 `--wait-for-consumers`，会等待至少一个 Receiver/Viewer 连接后再开始推流（避免高倍速时丢掉开头数据）。若要单独跑 Sender：加 `--no-wait-for-consumers`。

## Marker(JSON) 字段约定（ANN stream）

每条 marker 是一个 JSON 字符串，至少包含：

- `kind`: `"annotation" | "segment_start" | "segment_end" | "session_start" | "session_end"`
- `global_sample`: 全局样本索引（从整个播放会话开始累计，基于 `fs_out`）
- `symbol`: WFDB 符号（beat: `N/S/V/Q` 等；节律 token 用 `+`）
- `aux_note`: 仅当 `symbol == '+'` 时通常存在（节律 token：`'(N' '(AFIB' '(AFL' ')'`）

Receiver 写回 `.atr` 时使用这些字段，并按窗口把 `global_sample` 转成窗口相对 sample。
