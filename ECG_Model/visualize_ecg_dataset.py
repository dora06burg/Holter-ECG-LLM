#!/usr/bin/env python3
# 功能（数据集快速体检/可视化小工具）：
# - 遍历 WFDB 数据集目录（包含 RECORDS 或 .hea/.dat/.atr），统计每条记录的采样率、长度、时长、通道信息；
# - 输出：dataset_summary.csv（头信息汇总）、annotation_counts.csv（ann.symbol 计数）；
# - 生成若干图：时长直方图、采样率直方图、幅值分布直方图，以及随机抽取若干条记录的短窗口波形图（可叠加标注点）。
#
# 运行方法：
# - 例：`python ECG_Model/visualize_ecg_dataset.py --dataset ECG_Model/dataset/Icentia11k --outdir ECG_Model/viz_output`
# - 可选：`--num-records 6 --seconds 10 --seed 42 --max-ann 5000`
# - 依赖：`pip install wfdb matplotlib`
"""
中文说明
----------------------
这个脚本的定位是“数据集快速体检 + 小规模可视化”：

当你刚下载/整理好一份 WFDB 格式的 ECG 数据集时，常见问题包括：
- 我到底有多少条 record？
- 每条 record 采样率是多少？长度是多少？是不是都一致？
- 标注文件能不能读？里面有哪些符号？数量大概多少？
- 随便抽几条波形画出来，看看幅值范围、噪声、漂移、标注点大致位置是否合理。

一、WFDB 数据在磁盘上的典型结构
1) 每条 record 通常对应：
   - `xxx.hea`：头信息（文本）
   - `xxx.dat`：波形（二进制）
   - `xxx.atr`：标注（二进制，可选）
2) `RECORDS` 文件（文本）：
   - 每行一个 record 的“基名”（不带扩展名）
   - 有了它，就能可靠地遍历数据集，即使 record 分布在子目录中也没问题

二、这个脚本会输出什么？
输出目录（--outdir，默认 `ECG_Model/viz_output/`）会得到：
- `dataset_summary.csv`：每条 record 的 fs、长度、时长、通道数、单位等汇总
- `annotation_counts.csv`：把所有 record 的 ann.symbol 统计计数（读得到 .atr 的情况下）
- `hist_duration_seconds.png`：每条 record 时长直方图
- `hist_sampling_frequency.png`：采样率直方图
- `hist_amplitude.png`：幅值直方图（只抽样每条 record 的前几秒，避免读太大）
- `*_waveform.png`：随机抽取若干条 record，画出开头几秒钟波形，并把标注点（ann.sample）叠加出来

三、适合什么时候用？
- 你刚下载一份数据，想快速确认“文件齐不齐、能不能读、元信息是否合理”
- 你改了数据路径/结构，想确认脚本还能正确找到 record

ECG dataset visualization helper.

This script reads WFDB-format records (.hea/.dat/.atr) and produces:
1) Example waveform plots (with annotation markers if available)
2) Dataset summary CSV
3) Histogram plots for signal duration and sampling frequency
4) Amplitude distribution plot (from short samples)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from typing import List, Tuple

import numpy as np

try:
    import wfdb
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency: wfdb. Install with `pip install wfdb` and retry."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency: matplotlib. Install with `pip install matplotlib` and retry."
    ) from exc


def read_records_list(dataset_dir: str) -> List[str]:
    # 读取数据集里所有 record 的列表（record 名称不带扩展名）
    # 1) 优先读 RECORDS（最标准）
    # 2) 没有 RECORDS 时，递归扫描所有 .hea 文件作为兜底
    records_path = os.path.join(dataset_dir, "RECORDS")
    if os.path.isfile(records_path):
        with open(records_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    records: List[str] = []
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            if not name.endswith(".hea"):
                continue
            # record_name 要传给 wfdb.rdrecord/rdheader/rdann，形式是“去掉扩展名后的路径”
            # 如果数据在子目录中（如 p09/09000/p09000_s00.hea），我们需要存相对路径。
            rel = os.path.relpath(os.path.join(root, os.path.splitext(name)[0]), dataset_dir)
            records.append(rel)
    return sorted(set(records))


def resolve_dataset_root(dataset_dir: str) -> str:
    """
    自动寻找真正的“数据根目录”（即包含 RECORDS + WFDB 文件的目录）。

    很多人会把数据放成：
      ECG_Model/dataset/Icentia11k/RECORDS
    但运行时可能传的是：
      ECG_Model/dataset
    这个函数会帮你自动找到那一层。
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if os.path.isfile(os.path.join(dataset_dir, "RECORDS")):
        return dataset_dir

    common = os.path.join(dataset_dir, "Icentia11k")
    if os.path.isfile(os.path.join(common, "RECORDS")):
        return common

    try:
        subdirs = [
            os.path.join(dataset_dir, name)
            for name in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, name))
        ]
    except FileNotFoundError:
        return dataset_dir

    candidates = [d for d in subdirs if os.path.isfile(os.path.join(d, "RECORDS"))]
    if len(candidates) == 1:
        return candidates[0]
    return dataset_dir


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sample_records(records: List[str], k: int, seed: int) -> List[str]:
    if k >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, k)


def header_summary(
    dataset_dir: str, records: List[str]
) -> Tuple[List[dict], Counter]:
    # 对每条 record 读取头信息，并（如果存在）读取标注文件做一个符号统计。
    rows = []
    ann_counter: Counter = Counter()
    for rec in records:
        rec_path = os.path.join(dataset_dir, rec)
        # rdheader 只读 .hea，速度快，适合先做总体统计
        header = wfdb.rdheader(rec_path)
        fs = float(header.fs) if header.fs is not None else None
        sig_len = int(header.sig_len) if header.sig_len is not None else None
        duration = sig_len / fs if fs and sig_len else None
        row = {
            "record": rec,
            "fs": fs,
            "sig_len": sig_len,
            "duration_sec": duration,
            "n_sig": header.n_sig,
            "sig_name": ",".join(header.sig_name or []),
            "units": ",".join(header.units or []),
        }
        rows.append(row)

        ann_path = f"{rec_path}.atr"
        if os.path.isfile(ann_path):
            try:
                # rdann 读取 annotation；Icentia11k 常用 extension="atr"
                ann = wfdb.rdann(rec_path, "atr")
                # ann.symbol 是一个符号列表（N/S/V/Q/+ 等），这里统计一下有哪些符号、出现多少次
                ann_counter.update(ann.symbol)
            except Exception:
                # Skip unreadable annotation files but keep going
                continue
    return rows, ann_counter


def plot_histograms(outdir: str, summary_rows: List[dict]) -> None:
    # 把 summary 里的关键数字画成直方图，快速看分布是否“集中一致”或“明显异常”。
    durations = [row["duration_sec"] for row in summary_rows if row["duration_sec"]]
    fs_list = [row["fs"] for row in summary_rows if row["fs"]]

    if durations:
        plt.figure(figsize=(8, 4.5))
        plt.hist(durations, bins=20, color="#3A7CA5", edgecolor="white")
        plt.title("Signal Duration (seconds)")
        plt.xlabel("Seconds")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "hist_duration_seconds.png"), dpi=150)
        plt.close()

    if fs_list:
        plt.figure(figsize=(8, 4.5))
        plt.hist(fs_list, bins=10, color="#7A9E7E", edgecolor="white")
        plt.title("Sampling Frequency (Hz)")
        plt.xlabel("Hz")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "hist_sampling_frequency.png"), dpi=150)
        plt.close()


def plot_amplitude_distribution(
    dataset_dir: str,
    outdir: str,
    records: List[str],
    sample_seconds: float,
) -> None:
    # 画一个“幅值分布直方图”（只取每条 record 的前 sample_seconds 秒）
    # 为什么只取前几秒？
    # - 全数据可能很大，读全量会很慢
    # - 我们这里只是想快速了解一下幅值范围、是否有饱和/异常量程
    samples = []
    for rec in records:
        rec_path = os.path.join(dataset_dir, rec)
        header = wfdb.rdheader(rec_path)
        if not header.fs:
            continue
        sampto = int(header.fs * sample_seconds)
        record = wfdb.rdrecord(rec_path, sampto=sampto)
        # p_signal：物理单位（例如 mV）；d_signal：数字量化值（int）
        # 如果 p_signal 可用，一般优先用它（更直观）
        signal = record.p_signal if record.p_signal is not None else record.d_signal
        if signal is None:
            continue
        channel = signal[:, 0] if signal.ndim > 1 else signal
        samples.append(channel)

    if not samples:
        return

    stacked = np.concatenate(samples)
    plt.figure(figsize=(8, 4.5))
    plt.hist(stacked, bins=60, color="#C06C84", edgecolor="white")
    plt.title(f"Amplitude Distribution (first {sample_seconds}s of samples)")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_amplitude.png"), dpi=150)
    plt.close()


def plot_waveforms(
    dataset_dir: str,
    outdir: str,
    records: List[str],
    seconds: float,
    max_ann: int,
) -> None:
    # 随机抽取若干条 record，画出每条开头 seconds 秒的波形，方便你肉眼快速检查数据是否“像 ECG”。
    for rec in records:
        rec_path = os.path.join(dataset_dir, rec)
        header = wfdb.rdheader(rec_path)
        if not header.fs:
            continue
        sampto = int(header.fs * seconds)
        record = wfdb.rdrecord(rec_path, sampto=sampto)
        signal = record.p_signal if record.p_signal is not None else record.d_signal
        if signal is None:
            continue
        channel = signal[:, 0] if signal.ndim > 1 else signal
        time_axis = np.arange(channel.shape[0]) / header.fs

        plt.figure(figsize=(10, 4.5))
        plt.plot(time_axis, channel, color="#2F4858", linewidth=1.0, label="ECG")

        ann_path = f"{rec_path}.atr"
        if os.path.isfile(ann_path):
            try:
                ann = wfdb.rdann(rec_path, "atr")
                # ann.sample 是“采样点索引”，把它除以 fs 就得到秒
                ann_samples = ann.sample[:max_ann]
                mask = ann_samples < channel.shape[0]
                ann_samples = ann_samples[mask]
                if ann_samples.size:
                    plt.scatter(
                        ann_samples / header.fs,
                        channel[ann_samples],
                        color="#E76F51",
                        s=12,
                        label="Annotations",
                    )
            except Exception:
                pass

        plt.title(f"{rec} - first {seconds}s")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{rec}_waveform.png"), dpi=150)
        plt.close()


def write_summary_csv(outdir: str, rows: List[dict]) -> str:
    path = os.path.join(outdir, "dataset_summary.csv")
    fieldnames = [
        "record",
        "fs",
        "sig_len",
        "duration_sec",
        "n_sig",
        "sig_name",
        "units",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_annotation_summary(outdir: str, ann_counter: Counter) -> None:
    if not ann_counter:
        return
    path = os.path.join(outdir, "annotation_counts.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "count"])
        for symbol, count in ann_counter.most_common():
            writer.writerow([symbol, count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize WFDB ECG dataset with basic plots."
    )
    default_dataset = os.path.join(os.path.dirname(__file__), "dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Path to the dataset folder containing RECORDS and WFDB files.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "viz_output"),
        help="Directory to save generated plots and summaries.",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=6,
        help="Number of records to visualize with waveform plots.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Seconds to show per waveform plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling records.",
    )
    parser.add_argument(
        "--max-ann",
        type=int,
        default=5000,
        help="Max annotations to plot per record.",
    )
    parser.add_argument(
        "--amp-seconds",
        type=float,
        default=5.0,
        help="Seconds per record used for amplitude histogram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 兼容 dataset/Icentia11k 多一层目录的情况
    dataset_dir = resolve_dataset_root(args.dataset)
    outdir = os.path.abspath(args.outdir)
    ensure_dir(outdir)

    records = read_records_list(dataset_dir)
    if not records:
        raise SystemExit(f"No records found in {dataset_dir}")

    summary_rows, ann_counter = header_summary(dataset_dir, records)
    write_summary_csv(outdir, summary_rows)
    write_annotation_summary(outdir, ann_counter)

    plot_histograms(outdir, summary_rows)

    sample = sample_records(records, args.num_records, args.seed)
    plot_waveforms(dataset_dir, outdir, sample, args.seconds, args.max_ann)
    plot_amplitude_distribution(dataset_dir, outdir, sample, args.amp_seconds)

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
