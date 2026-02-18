#!/usr/bin/env python3
# 功能（按“节律区间/病症”可视化）：
# - 面向 Icentia11k 的 WFDB 数据（.hea/.dat/.atr），读取 .atr 中 ann.aux_note 编码的节律区间：
#   (N ... )、(AFIB ... )、(AFL ... ) 等；
# - 把同一患者的多个 segment（pXXXXX_sYY）按顺序拼接后，输出一张“节律总览图”：
#   1) ECG 长程包络 + 节律区间染色；2) 单行节律时间轴；3) 各节律总时长统计；
# - 用虚线标出 segment 边界，便于你判断拼接处（提示：子集数据可能不连续）。
#
# 运行方法：
# - 例：`python ECG_Model/plot_icentia11k_rhythm_overview.py --dataset ECG_Model/dataset --patient 0`
# - 如你的数据在子目录：`python ECG_Model/plot_icentia11k_rhythm_overview.py --dataset ECG_Model/dataset/Icentia11k --patient 0`
# - 常用参数：`--window-sec 1`（包络窗口秒数）、`--out 输出路径`
"""
中文说明（给代码初学者）
----------------------
这个脚本专门用来可视化 Icentia11k 的“节律区间（rhythm intervals）”标注，也就是你说的
按病症/节律区间标注（NSR/AFib/AFlutter）的那部分。

一、你需要知道的 WFDB 基础
Icentia11k 使用 WFDB 格式组织数据。一个 record（也就是一个 segment，约 70 分钟）通常包含：
- `xxx.hea`：头文件（文本），例如采样率 fs=250Hz、信号长度 sig_len 等
- `xxx.dat`：波形数据（二进制）
- `xxx.atr`：标注数据（二进制，WFDB annotation 格式）

二、节律区间标注在哪里？
节律区间标注在 `.atr` 中，通过 `wfdb.rdann(record, "atr")` 读取，主要看两个字段：
1) `ann.aux_note`：一串 token（事件流），例如：
   - `'(N'` 表示“从这个采样点开始进入 NSR”
   - `'(AFIB'` 表示“从这个采样点开始进入 AFib”
   - `'(AFL'` 表示“从这个采样点开始进入 AFlutter”
   - `')'` 表示“结束最近一次开始的节律区间”
2) `ann.sample`：这些 token 对应发生的采样点索引（相对该 segment 的起点）

要把 token 变成真正的“区间”，就要做一个“括号配对”：
- 看到 `'(AFIB'` 记住开始位置
- 看到 `')'` 就把上次开始的位置拿出来，形成 [start, end] 区间

三、这个脚本做了什么（处理流程）
1) 找到某个患者的所有 segment（pXXXXX_sYY），按段号排序
2) 计算每段在“拼接后的长序列”中的 sample 偏移 start_sample_global
3) 逐段解析 `.atr`，把节律 token 配对成一组节律区间（全局 sample）
4) 生成 3 个视图：
   - Panel 1：ECG 长程包络 + 节律区间染色（帮助你对齐波形趋势与节律）
   - Panel 2：单行节律时间轴（最直观地展示什么时候是 AFIB/AFL/NSR）
   - Panel 3：每种节律的总时长（小时）统计

四、关于“拼接后是否连续”
公开子集常会随机抽取若干段 segment（例如 50 段），所以拼接后的时间线不一定等于真实佩戴的连续时间线。
因此图中会用虚线标注 segment 边界，提醒你注意拼接点。

Icentia11k (WFDB) rhythm-interval visualization.

Goal
----
Visualize *rhythm interval* annotations (NSR / AFIB / AFL) that are encoded in
WFDB annotation files (.atr) via `ann.aux_note`.

Encoding recap (as described by the dataset):
  - Beat labels: ann.symbol at ann.sample positions (N, S, V, Q, ...)
  - Rhythm labels: ann.aux_note tokens:
        '(N' / '(AFIB' / '(AFL'  start a rhythm region
        ')'                   ends the most recent region

This script concatenates all segments for a patient (sorted by segment id)
and produces a rhythm-focused overview figure:
  1) Long ECG envelope with rhythm shading
  2) A single-row rhythm timeline bar (intervals)
  3) Total duration per rhythm label

Important note about continuity
-------------------------------
Public Icentia11k releases often provide a subset (e.g., 50 randomly selected
segments per patient). Concatenation produces a long signal but may not reflect
a truly continuous wearable timeline; segment boundaries are drawn as dashed lines.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import wfdb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: wfdb. Install with `pip install wfdb`.") from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: matplotlib. Install with `pip install matplotlib`."
    ) from exc


RE_RECORD_BASENAME = re.compile(r"^p(?P<pid>\d{5})_s(?P<sid>\d+)$")


@dataclass(frozen=True)
class SegmentInfo:
    # SegmentInfo 保存“这一段 segment 的元信息”，后续需要用它来：
    # - 给节律区间加上全局偏移（拼接）
    # - 把 sample 换算成秒/小时（fs）
    record: str  # record name relative to dataset dir (no extension)
    patient_id: int
    segment_id: int
    fs: float
    sig_len: int
    start_sample_global: int

    @property
    def start_sec_global(self) -> float:
        return float(self.start_sample_global) / float(self.fs)


def read_records_list(dataset_dir: str) -> List[str]:
    # 读取 RECORDS 列表（最标准），如果没有就从 .hea 文件推断 record 名称作为兜底。
    records_path = os.path.join(dataset_dir, "RECORDS")
    if os.path.isfile(records_path):
        with open(records_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    records: List[str] = []
    for name in os.listdir(dataset_dir):
        if name.endswith(".hea"):
            records.append(os.path.splitext(name)[0])
    return sorted(set(records))


def resolve_dataset_root(dataset_dir: str) -> str:
    """
    Resolve the folder that actually contains WFDB files + RECORDS.

    Some local layouts place the data under an extra subfolder, e.g.:
      ECG_Model/dataset/Icentia11k/RECORDS
    while callers may pass:
      ECG_Model/dataset
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
    if len(candidates) > 1:
        raise SystemExit(
            f"Multiple dataset roots found under {dataset_dir}: {candidates}. "
            "Please pass --dataset pointing to the folder containing RECORDS."
        )
    return dataset_dir


def parse_patient_and_segment(record: str) -> Optional[Tuple[int, int]]:
    # 从 record 名字解析 patient_id 与 segment_id。
    # record 可能带子目录，所以取 basename 后匹配：p00000_s01 => pid=0, sid=1
    base = os.path.basename(record)
    match = RE_RECORD_BASENAME.match(base)
    if not match:
        return None
    return int(match.group("pid")), int(match.group("sid"))


def available_patients(records: Sequence[str]) -> List[int]:
    pids = set()
    for rec in records:
        parsed = parse_patient_and_segment(rec)
        if parsed is None:
            continue
        pid, _ = parsed
        pids.add(pid)
    return sorted(pids)


def build_patient_segments(
    dataset_dir: str, records: Sequence[str], patient_id: int
) -> List[SegmentInfo]:
    # 收集某个患者的所有段，并按段号排序，同时计算每段在拼接序列里的起始偏移 start_sample_global。
    patient_records: List[Tuple[int, str]] = []
    for rec in records:
        parsed = parse_patient_and_segment(rec)
        if parsed is None:
            continue
        pid, sid = parsed
        if pid == patient_id:
            patient_records.append((sid, rec))

    if not patient_records:
        raise SystemExit(f"No records found for patient p{patient_id:05d}.")

    patient_records.sort(key=lambda x: x[0])

    segments: List[SegmentInfo] = []
    start_sample_global = 0
    fs_ref: Optional[float] = None
    for sid, rec in patient_records:
        # 读取头信息（.hea）即可得到 fs/sig_len，不需要先读波形。
        header = wfdb.rdheader(os.path.join(dataset_dir, rec))
        if header.fs is None or header.sig_len is None:
            raise SystemExit(f"Missing fs/sig_len in header for record: {rec}")
        fs = float(header.fs)
        sig_len = int(header.sig_len)
        # 同一患者内部采样率应一致，否则 sample->time 的换算会错。
        if fs_ref is None:
            fs_ref = fs
        elif abs(fs_ref - fs) > 1e-6:
            raise SystemExit(
                f"Inconsistent sampling rate within patient: {fs_ref} vs {fs}"
            )
        segments.append(
            SegmentInfo(
                record=rec,
                patient_id=patient_id,
                segment_id=sid,
                fs=fs,
                sig_len=sig_len,
                start_sample_global=start_sample_global,
            )
        )
        start_sample_global += sig_len
    return segments


def _safe_aux(aux: str) -> str:
    # 把 None / "None" 统一处理成空字符串，方便后面判断“是否有 token”。
    if aux is None:
        return ""
    if aux == "None":
        return ""
    return aux


def extract_rhythm_regions(
    dataset_dir: str, segments: Sequence[SegmentInfo]
) -> List[Tuple[int, int, str]]:
    """
    Return rhythm regions as (start_sample_global, end_sample_global, label).
    """
    # 输出采用“全局 sample 坐标”（拼接后），方便跨 segment 画在同一条时间轴上。
    # 其中 start/end 是采样点索引，label 是节律名（N/AFIB/AFL...）。
    regions: List[Tuple[int, int, str]] = []

    for seg in segments:
        rec_path = os.path.join(dataset_dir, seg.record)
        # 读取标注（.atr）。节律 token 在 ann.aux_note 中。
        ann = wfdb.rdann(rec_path, "atr")

        # stack（栈）用于配对 "(XXX" 和 ")"，把 token 事件流变成区间。
        stack: List[Tuple[str, int]] = []
        for sample, aux in zip(ann.sample, ann.aux_note):
            aux = _safe_aux(aux)
            if not aux:
                continue
            g_sample = seg.start_sample_global + int(sample)

            if aux.startswith("("):
                label = aux[1:]
                if label.endswith(")"):
                    # Instantaneous marker like "(N)"; not an interval.
                    continue

                # Close any currently open interval if a new one begins (defensive).
                # 防御性处理：如果遇到新的开始 token，但之前的区间还没闭合，
                # 就把旧区间强制在当前点闭合，避免解析结果乱掉。
                if stack:
                    prev_label, prev_start = stack.pop()
                    if g_sample > prev_start:
                        regions.append((prev_start, g_sample, prev_label))
                stack.append((label, g_sample))
                continue

            if aux == ")":
                if stack:
                    label, start_sample = stack.pop()
                    if g_sample > start_sample:
                        regions.append((start_sample, g_sample, label))

        # Close any unclosed region at the end of this segment.
        # 如果 segment 结束时仍有未闭合区间，则闭合到 segment 末尾。
        if stack:
            end_sample = seg.start_sample_global + seg.sig_len
            while stack:
                label, start_sample = stack.pop()
                if end_sample > start_sample:
                    regions.append((start_sample, end_sample, label))

    regions.sort(key=lambda x: (x[0], x[1], x[2]))
    return _merge_adjacent_regions(regions)


def _merge_adjacent_regions(
    regions: Sequence[Tuple[int, int, str]], gap_samples: int = 0
) -> List[Tuple[int, int, str]]:
    """
    Merge regions with the same label when they are adjacent or separated by a tiny gap.
    """
    # 有些情况下，同一标签的区间可能会被拆成多个紧邻的小区间；
    # 这里把相同标签且相邻的区间合并，方便画图更“干净”。
    if not regions:
        return []

    merged: List[Tuple[int, int, str]] = []
    cur_start, cur_end, cur_label = regions[0]
    for start, end, label in regions[1:]:
        if label == cur_label and start <= cur_end + gap_samples:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end, cur_label))
        cur_start, cur_end, cur_label = start, end, label
    merged.append((cur_start, cur_end, cur_label))
    return merged


def compute_envelope(
    dataset_dir: str, segments: Sequence[SegmentInfo], window_sec: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a global min/max envelope (in physical units) with a fixed window.
    Returns: (t_sec, y_min, y_max) arrays concatenated across segments.
    """
    if not segments:
        raise ValueError("No segments given")

    fs = segments[0].fs
    # window_sec（秒）-> window_samples（采样点数）
    window_samples = int(round(window_sec * fs))
    if window_samples < 1:
        raise ValueError("window_sec too small")

    t_chunks: List[np.ndarray] = []
    min_chunks: List[np.ndarray] = []
    max_chunks: List[np.ndarray] = []

    for seg in segments:
        rec_path = os.path.join(dataset_dir, seg.record)
        # 读取波形（只取通道 0）。physical=True 时 rec.p_signal 为物理单位（mV）
        rec = wfdb.rdrecord(rec_path, channels=[0])
        if rec.p_signal is None:
            raise SystemExit(f"Missing p_signal for record: {seg.record}")
        signal = rec.p_signal[:, 0].astype(np.float32, copy=False)
        n_full = signal.shape[0] // window_samples
        if n_full <= 0:
            continue

        trimmed = signal[: n_full * window_samples]
        windows = trimmed.reshape(n_full, window_samples)
        y_min = windows.min(axis=1)
        y_max = windows.max(axis=1)

        centers = (np.arange(n_full, dtype=np.float32) * window_samples + window_samples / 2) / fs
        t_sec = centers + seg.start_sec_global

        t_chunks.append(t_sec)
        min_chunks.append(y_min)
        max_chunks.append(y_max)

    if not t_chunks:
        raise SystemExit("No envelope points computed.")

    return np.concatenate(t_chunks), np.concatenate(min_chunks), np.concatenate(max_chunks)


def label_to_display(label: str) -> str:
    mapping = {
        "N": "NSR",
        "AFIB": "AFib",
        "AFL": "AFlutter",
    }
    return mapping.get(label, label)


def rhythm_colors() -> Dict[str, str]:
    return {
        "N": "#3A7CA5",  # blue
        "AFIB": "#E76F51",  # orange/red
        "AFL": "#9B5DE5",  # purple
        "OTHER": "#9AA0A6",  # gray
    }


def color_for_label(label: str) -> str:
    colors = rhythm_colors()
    if label in colors:
        return colors[label]
    return colors["OTHER"]


def plot_rhythm_overview(
    dataset_dir: str,
    segments: Sequence[SegmentInfo],
    out_path: str,
    window_sec: float,
    dpi: int,
) -> str:
    # 1) 把节律 token 解析成区间（全局 sample）
    regions = extract_rhythm_regions(dataset_dir, segments)
    # 2) 计算 ECG 长程包络（避免直接画几十小时的原始采样点）
    t_sec, y_min, y_max = compute_envelope(dataset_dir, segments, window_sec=window_sec)

    patient_id = segments[0].patient_id
    fs = segments[0].fs
    total_samples = segments[-1].start_sample_global + segments[-1].sig_len
    total_sec = total_samples / fs
    total_hours = total_sec / 3600.0

    # 统计每种节律：
    # - dur_by_label：总持续采样点数（之后换算成小时）
    # - count_by_label：区间数量（多少段 AFIB/NSR 区间）
    dur_by_label: Counter = Counter()
    count_by_label: Counter = Counter()
    for start, end, label in regions:
        dur_by_label[label] += max(0, end - start)
        count_by_label[label] += 1

    # 画“区间条”需要 broken_barh 的输入格式：
    # - 每个条用 (start_x, width) 表示，这里 x 单位使用小时（更适合长程）
    bars_by_label: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
    for start, end, label in regions:
        start_h = (start / fs) / 3600.0
        dur_h = ((end - start) / fs) / 3600.0
        if dur_h <= 0:
            continue
        bars_by_label[label].append((start_h, dur_h))

    # Figure
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.7, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0])

    # Panel 1: ECG envelope
    # Panel 1：包络图 + 节律区间染色（帮助把“节律”与“波形趋势”对齐）
    ax0.fill_between(
        t_sec / 3600.0,
        y_min,
        y_max,
        color="#1D3557",
        alpha=0.55,
        linewidth=0.0,
        label=f"ECG envelope (min/max per {window_sec:g}s)",
    )

    # Rhythm shading on envelope
    # 根据不同 label，用不同颜色把区间染色
    for label, bars in bars_by_label.items():
        col = color_for_label(label)
        for start_h, dur_h in bars:
            ax0.axvspan(
                start_h,
                start_h + dur_h,
                color=col,
                alpha=0.18,
                linewidth=0,
            )

    # Segment boundaries
    # segment 边界虚线：提醒拼接点在哪里（子集数据不一定连续）
    for seg in segments[1:]:
        ax0.axvline(seg.start_sec_global / 3600.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)

    ax0.set_title(
        f"Icentia11k rhythm overview | patient p{patient_id:05d} | "
        f"{len(segments)} segments | {total_hours:.2f} h (concatenated) | fs={fs:g}Hz"
    )
    ax0.set_xlabel("Time (hours, concatenated segments)")
    ax0.set_ylabel("ECG (mV)")
    ax0.grid(True, alpha=0.2)

    # Panel 2: rhythm timeline bar
    # Panel 2：单行节律时间轴（最直观的“病症区间图”）
    y0, height = 0, 10
    for label, bars in bars_by_label.items():
        ax1.broken_barh(
            bars,
            (y0, height),
            facecolors=color_for_label(label),
            edgecolors="none",
            alpha=0.95,
            label=f"{label_to_display(label)} ({count_by_label[label]} regions)",
        )

    for seg in segments[1:]:
        ax1.axvline(seg.start_sec_global / 3600.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)

    ax1.set_ylim(0, 10)
    ax1.set_yticks([])
    ax1.set_title("Rhythm intervals (from ann.aux_note)")
    ax1.set_xlabel("Time (hours, concatenated segments)")
    ax1.grid(True, axis="x", alpha=0.25)

    # Legend (unique)
    handles, labels = ax1.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        ax1.legend(list(uniq.values()), list(uniq.keys()), loc="upper right", frameon=True)

    # Panel 3: total duration per rhythm
    # Panel 3：每种节律的总时长统计（小时 + 百分比）
    labels_sorted = sorted(dur_by_label.keys(), key=lambda k: dur_by_label[k], reverse=True)
    if labels_sorted:
        dur_hours = [dur_by_label[l] / fs / 3600.0 for l in labels_sorted]
        colors = [color_for_label(l) for l in labels_sorted]
        ax2.bar([label_to_display(l) for l in labels_sorted], dur_hours, color=colors, edgecolor="white")
        ax2.set_title("Total duration per rhythm label")
        ax2.set_ylabel("Hours")
        ax2.grid(True, axis="y", alpha=0.25)
        for i, (l, h) in enumerate(zip(labels_sorted, dur_hours)):
            pct = 0.0 if total_sec <= 0 else (h / total_hours * 100.0)
            ax2.text(i, h, f"{h:.2f}h\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    else:
        ax2.text(0.01, 0.5, "No rhythm regions parsed from aux_note.", transform=ax2.transAxes, ha="left", va="center")
        ax2.set_axis_off()

    # Add a compact legend for colors (in case only one label exists)
    patches = []
    for key in ["N", "AFIB", "AFL", "OTHER"]:
        patches.append(Patch(facecolor=color_for_label(key), label=label_to_display(key)))
    ax0.legend(handles=patches, loc="upper right", frameon=True, title="Rhythm color map")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot rhythm interval annotations (aux_note) for an Icentia11k patient."
    )
    default_dataset = os.path.join(os.path.dirname(__file__), "dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Path to the dataset folder containing RECORDS and WFDB files.",
    )
    parser.add_argument(
        "--patient",
        type=int,
        default=None,
        help="Patient ID (e.g., 0 for p00000). If omitted and only one patient exists, it is used.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (.png). Default: ECG_Model/viz_output/icentia_pXXXXX_rhythm_overview.png",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Envelope window size in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150).",
    )
    return parser.parse_args()


def main() -> None:
    # 兼容某些环境里 numpy longdouble dtype 的警告输出
    warnings.filterwarnings("ignore", category=UserWarning, message=".*broken support for the dtype.*")

    args = parse_args()
    # 兼容 dataset/Icentia11k 这种多一层目录的情况
    dataset_dir = resolve_dataset_root(args.dataset)
    records = read_records_list(dataset_dir)
    if not records:
        raise SystemExit(f"No WFDB records found in: {dataset_dir}")

    pids = available_patients(records)
    patient_id = args.patient
    if patient_id is None:
        if len(pids) == 1:
            patient_id = pids[0]
        else:
            raise SystemExit(
                f"Multiple patients found ({len(pids)}). Please pass --patient. Example: --patient {pids[0]}"
            )
    if patient_id not in pids:
        raise SystemExit(
            f"Patient p{patient_id:05d} not found. Available: {pids[:10]}{'...' if len(pids)>10 else ''}"
        )

    segments = build_patient_segments(dataset_dir, records, patient_id=patient_id)
    out_path = args.out
    if out_path is None:
        outdir = os.path.join(os.path.dirname(__file__), "viz_output")
        out_path = os.path.join(outdir, f"icentia_p{patient_id:05d}_rhythm_overview.png")

    saved = plot_rhythm_overview(
        dataset_dir=dataset_dir,
        segments=segments,
        out_path=os.path.abspath(out_path),
        window_sec=float(args.window_sec),
        dpi=int(args.dpi),
    )
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
