# Icentia11k 工具集（WFDB 可视化 + LSL 流式模拟）

## 概述
本目录包含一组面向 **Icentia11k（WFDB 格式）** 的小工具，主要用途：
- 快速体检/可视化数据集（统计 + 图像输出）。
- 长程 ECG 拼接可视化与节律区间总览。
- LSL 流式模拟（发送/接收/实时查看）与对齐检查。

## 目录结构
```text
Icentia11k_code/
├─ plot_icentia11k_long_ecg.py
├─ plot_icentia11k_rhythm_overview.py
├─ plot_stream_wfdb_alignment.py
├─ visualize_ecg_dataset.py
├─ code_for_stream/
│  ├─ sender_icentia_lsl.py
│  ├─ receiver_lsl_to_wfdb.py
│  ├─ viewer_lsl_realtime.py
│  ├─ viewer_lsl_realtime_live.py
│  ├─ plot_wfdb_windows.py
│  ├─ icentia_wfdb.py
│  ├─ lsl_utils.py
│  ├─ generate_stream_result.py
│  ├─ requirements.txt
│  ├─ ONBOARDING.md
│  ├─ README.md
│  └─ vendor/
├─ stream_result/
│  ├─ lsl_wfdb_p00000_s00_s49_speed500/
│  ├─ run_20260209_163829/
│  ├─ run_20260210_092709/
│  └─ report/
└─ viz_output/
```

## 脚本功能概述
- `visualize_ecg_dataset.py`
  - 数据集“体检”脚本：统计采样率/时长/通道信息，输出 `dataset_summary.csv`、`annotation_counts.csv`，并生成直方图与随机波形图。
- `plot_icentia11k_long_ecg.py`
  - 按患者拼接多个 segment，生成长程包络 + 非正常 beat 时间线 + 局部放大窗口。
- `plot_icentia11k_rhythm_overview.py`
  - 解析 `.atr` 中节律区间（NSR/AFIB/AFL），输出节律总览图与各节律总时长统计。
- `plot_stream_wfdb_alignment.py`
  - 对齐检查工具：在 LSL 切窗写回的 WFDB 记录中评估标注点与波形峰值的一致性。

## 重要子目录说明
- `code_for_stream/`
  - LSL 流式模拟与回写工具集合，包含 Sender/Receiver/Viewer 以及辅助脚本。
  - 详细使用方法见 `code_for_stream/README.md` 与 `code_for_stream/ONBOARDING.md`。
- `stream_result/`
  - 流式实验输出（WFDB 切窗结果、报告与日志）。
- `viz_output/`
  - 本目录脚本生成的图像与 CSV 输出。

## 运行环境
- Python 依赖写在本目录的 `requirements.txt`。
- 若使用 LSL 流式功能，`code_for_stream/vendor/` 已提供可用的 `liblsl` 动态库（详见 `code_for_stream/README.md`）。
