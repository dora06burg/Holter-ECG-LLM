# Icentia11k LSL 流式模拟：运行结果

本报告由 `ECG_Model/code_for_stream/generate_stream_result.py` 自动生成。

## 目录结构

- `wfdb_250Hz/`：Receiver 写回的 WFDB 窗口（`.dat/.hea/.atr` + `RECORDS`）

- `report/assets/`：窗口波形 + 标注可视化 PNG

- `logs/`：sender/receiver 的运行日志


## 本次运行参数（关键）

```json
{
  "run_id": "20260210_092709",
  "patient": 0,
  "window_sec": 10.0,
  "hop_sec": 10.0,
  "chunk_ms": 200.0,
  "inject_rhythm_state": true,
  "demos": [
    {
      "name": "250Hz",
      "fs_out": 250.0,
      "seconds": 30.0,
      "windows": 3,
      "speed": 20.0
    }
  ],
  "resample_demo": false,
  "paths": {
    "run_dir": "/mnt/e/2025grade3/code/ECG_Model/stream_result/run_20260210_092709",
    "wfdb_250Hz": "/mnt/e/2025grade3/code/ECG_Model/stream_result/run_20260210_092709/wfdb_250Hz",
    "wfdb_200Hz": "",
    "report_dir": "/mnt/e/2025grade3/code/ECG_Model/stream_result/run_20260210_092709/report",
    "logs_dir": "/mnt/e/2025grade3/code/ECG_Model/stream_result/run_20260210_092709/logs"
  }
}
```

## 可视化结果（示例）

### 250Hz 窗口

下面每张图对应一个 10 秒窗口，背景色表示节律（NSR/AFIB/AFL），散点表示 beat 标注。

（注意：NSR 的起始 token 在 Icentia11k 里通常是 `+( '(N')`，报告里会显示为 NSR。）


![](assets/250Hz/summary.png)

## 快速上手（你自己复现）

推荐按以下顺序启动，避免高倍速时丢开头：

1. （可选）实时 Viewer：`python ECG_Model/code_for_stream/viewer_lsl_realtime.py`

2. Receiver（写回 WFDB）：`python ECG_Model/code_for_stream/receiver_lsl_to_wfdb.py --out-dir ... --inject-rhythm-state --verbose`

3. Sender（播放并推流）：`python ECG_Model/code_for_stream/sender_icentia_lsl.py --dataset ECG_Model/dataset --patient 0 --speed 20 --verbose`


更详细说明见：`ECG_Model/code_for_stream/README.md`。
