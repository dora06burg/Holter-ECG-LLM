#!/usr/bin/env python3
# 功能（长程拼接 + 可视化）：
# - 面向 Icentia11k 的 WFDB 数据（.hea/.dat/.atr），按“同一患者”的多个 segment（pXXXXX_sYY）顺序拼接成长程信号；
# - 生成一张总览图：全程 ECG 包络（降采样后的 min/max envelope）+ 非正常 beat（Q/S/V）的时间线 + 1~2 个局部放大窗口；
# - 额外用黑色虚线标出 segment 边界（提示：公开子集常是随机抽取的 50 段，拼接后不一定代表真实连续时间）。
#
# 运行方法：
# - 例：`python ECG_Model/plot_icentia11k_long_ecg.py --dataset ECG_Model/dataset --patient 0`
# - 如你的数据在子目录：`python ECG_Model/plot_icentia11k_long_ecg.py --dataset ECG_Model/dataset/Icentia11k --patient 0`
# - 常用参数：`--window-sec 1`（包络窗口秒数）、`--zoom-half-sec 4`（放大图半窗口秒数）、`--out 输出路径`
"""
中文说明（给代码初学者）
----------------------
这个脚本用于把 Icentia11k（WFDB 格式）的 ECG 数据做“长程可视化”（long-range view）。

一、Icentia11k 数据在磁盘上的基本形态
1) 一条“记录/record”（也是一段 segment）通常由 3 个文件组成（同名不同扩展名）：
   - `pXXXXX_sYY.hea`：头文件（文本），包含采样率 fs、采样点数 sig_len、单位/增益等
   - `pXXXXX_sYY.dat`：波形数据（二进制），通常是 int16
   - `pXXXXX_sYY.atr`：标注文件（二进制，WFDB annotation 格式）
2) `RECORDS`：文本文件，每行是一个 record 的“基名”（不带扩展名），用于遍历数据集。
3) Icentia11k 常见命名：
   - 患者 ID：`p00000`
   - 第 0 段：`p00000_s00`
   - 第 1 段：`p00000_s01`
   - ...

二、标注（annotation）在哪里？
这份数据的“beat 标注 + 节律区间标注”都在 `.atr` 里：
1) Beat（逐心搏）标注：
   - `ann.sample`：某个心搏（R 峰）所在的采样点索引（相对该 segment 起点）
   - `ann.symbol`：该心搏的类别（N/S/V/Q 等）
2) Rhythm（节律区间）标注：
   - `ann.aux_note`：一串 token 事件流
     - `'(N' / '(AFIB' / '(AFL'` 表示“从这里开始进入某种节律”
     - `')'` 表示“结束最近一次开始的节律区间”
   - 解析时需要把这些 start/end token “配对”成区间。

三、这个脚本做了什么（处理流程）
因为每段 segment 很长（约 70 分钟），直接画完整波形会非常密集、难以观察全局趋势。
所以这里使用“包络图（envelope）”来压缩显示：
1) 找到某个患者的所有 segment（按 s00、s01…排序）
2) 计算每个 segment 在“拼接后长程信号”中的起始偏移 start_sample_global
3) 逐段读取波形，用固定窗口（默认 1 秒）计算每窗的 min/max，形成包络
4) 逐段读取 `.atr`：
   - 收集 beat 的全局时间（把 sample + 偏移，然后除以 fs 变成秒）
   - 解析节律区间并画成背景色块
5) 输出一张总览图：
   - Panel 1：长程包络 + 节律区间染色 + segment 边界虚线
   - Panel 2：非正常 beat（V/S/Q）的时间线 + 节律区间染色
   - Panel 3/4：围绕少见/非正常 beat 的局部放大图（便于看细节波形）

四、重要提醒：拼接 ≠ 一定连续
Icentia11k 的公开子集常会“随机抽取 50 段 segment”，因此把 s00..s49 拼起来
能得到一条很长的可视化/训练序列，但它不一定对应原始佩戴记录中的完全连续时间线。
为了提醒你这一点，我们在图里画了 segment 边界虚线。

Icentia11k (WFDB) long-range visualization.

This helper focuses on the Icentia11k encoding conventions:
- Each record is a single-lead ECG segment, typically ~70 minutes.
- Beat annotations live in the ".atr" file:
    - ann.sample: sample index of the beat (R peak timepoint)
    - ann.symbol: beat class (N, S, V, Q, ...)
    - ann.symbol == '+' indicates a rhythm annotation at this timepoint
- Rhythm annotations are encoded in ann.aux_note:
    - '(N', '(AFIB', '(AFL' start a region
    - ')' ends the most recent region

For a *long-range* view, this script concatenates all segments for a given
patient (by sorting on the segment index in the filename) and produces:
1) A global min/max envelope plot (per-second by default) of the ECG
2) A compact timeline of non-normal beat markers (e.g., Q/S/V) and rhythm regions
3) One or two zoom-in windows around rare/non-normal beats (if present)

Notes
-----
The public Icentia11k releases commonly include a *subset* of segments per
patient (e.g., 50 randomly selected segments). In that case, concatenating
segments will produce a long signal, but it may not reflect a truly continuous
recording (there can be gaps in the original timeline). Segment boundaries are
drawn as vertical dashed lines.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import wfdb
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency: wfdb. Install with `pip install wfdb`.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency: matplotlib. Install with `pip install matplotlib`."
    ) from exc


RE_RECORD_BASENAME = re.compile(r"^p(?P<pid>\d{5})_s(?P<sid>\d+)$")


@dataclass(frozen=True)
class SegmentInfo:
    # 这是一段 segment 的“元信息”（不是波形本身）。
    # 我们之所以把它单独封装起来，是因为后面拼接与画图时需要：
    # - 这一段是哪个 record（文件名）
    # - fs 采样率是多少
    # - 这一段有多少采样点 sig_len
    # - 这一段在“拼接后的长程信号”里从哪个 sample 开始（start_sample_global）
    record: str  # WFDB record path relative to dataset_dir (no extension)
    patient_id: int
    segment_id: int
    fs: float
    sig_len: int
    start_sample_global: int

    @property
    def duration_sec(self) -> float:
        return float(self.sig_len) / float(self.fs)

    @property
    def start_sec_global(self) -> float:
        return float(self.start_sample_global) / float(self.fs)


@dataclass(frozen=True)
class ZoomRequest:
    # ZoomRequest 用来描述“需要放大查看的窗口”应该取哪里：
    # - 哪个 record
    # - record 内的哪个 sample 附近
    # - 想重点看的 beat 类型（N/Q/S/V）
    title: str
    record: str
    sample_in_record: int
    fs: float
    kind: str


def read_records_list(dataset_dir: str) -> List[str]:
    # 读取记录列表（record 名称，不带扩展名）。
    # 优先使用 RECORDS 文件（最标准、最快、也适合含子目录的结构）。
    # 如果没有 RECORDS，就递归扫描 .hea 文件作为兜底。
    records_path = os.path.join(dataset_dir, "RECORDS")
    if os.path.isfile(records_path):
        with open(records_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    records: List[str] = []
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            if not name.endswith(".hea"):
                continue
            rel = os.path.relpath(os.path.join(root, os.path.splitext(name)[0]), dataset_dir)
            records.append(rel)
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
    # 从 record 名字里解析出 (patient_id, segment_id)。
    # 例如：p09000_s03 => (9000, 3)
    #
    # 注意：在完整数据集里 record 可能带有子目录（如 p09/09000/p09000_s00），
    # 所以这里用 basename 取最后一段文件名再做正则匹配。
    base = os.path.basename(record)
    match = RE_RECORD_BASENAME.match(base)
    if not match:
        return None
    return int(match.group("pid")), int(match.group("sid"))


def available_patients(records: Sequence[str]) -> List[int]:
    # 从 records 列表中提取有哪些 patient_id 可用（去重后排序）。
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
    # 构建指定患者的 segment 列表（按 segment_id 排序），并计算每段在拼接序列中的起始 sample 偏移。
    #
    # start_sample_global 的计算方式：
    # - 第 0 段 start=0
    # - 第 1 段 start=第 0 段 sig_len
    # - 第 2 段 start=第 0 段 sig_len + 第 1 段 sig_len
    # - ...
    #
    # 这样可以把“段内 sample（0..sig_len-1）”映射到“全局 sample（拼接后）”：
    #   global_sample = segment_start_sample_global + sample_in_segment
    patient_records: List[Tuple[int, str]] = []
    for rec in records:
        parsed = parse_patient_and_segment(rec)
        if parsed is None:
            continue
        pid, sid = parsed
        if pid != patient_id:
            continue
        patient_records.append((sid, rec))

    if not patient_records:
        raise SystemExit(f"No records found for patient p{patient_id:05d} in {dataset_dir}")

    patient_records.sort(key=lambda x: x[0])

    segments: List[SegmentInfo] = []
    start_sample_global = 0
    fs_ref: Optional[float] = None

    for sid, rec in patient_records:
        # 读取头文件（.hea）获取 fs 和 sig_len（不需要加载整段波形，速度快）
        header = wfdb.rdheader(os.path.join(dataset_dir, rec))
        if header.fs is None or header.sig_len is None:
            raise SystemExit(f"Missing fs/sig_len in header for record: {rec}")

        fs = float(header.fs)
        sig_len = int(header.sig_len)
        # 同一个患者的所有段，采样率应该一致；这里做一个一致性检查，避免拼接时间轴出错。
        if fs_ref is None:
            fs_ref = fs
        elif abs(fs_ref - fs) > 1e-6:
            raise SystemExit(
                f"Inconsistent sampling rate for patient p{patient_id:05d}: {fs_ref} vs {fs}"
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
    # 在这个数据里，很多 beat 的 aux_note 会是字符串 "None"（不是 Python 的 None）。
    # 为了统一处理，我们把 None/"None"/空字符串都当作“没有节律 token”。
    if aux is None:
        return ""
    if aux == "None":
        return ""
    return aux


def extract_annotations(
    dataset_dir: str, segments: Sequence[SegmentInfo]
) -> Tuple[DefaultDict[str, List[float]], List[Tuple[float, float, str]], Counter]:
    """
    Returns:
      beat_times_by_symbol: {symbol: [time_sec_global, ...]}
      rhythm_regions: [(start_sec, end_sec, label), ...]
      beat_symbol_counts: Counter over all beat symbols (including '+')
    """
    beat_times_by_symbol: DefaultDict[str, List[float]] = defaultdict(list)
    rhythm_regions: List[Tuple[float, float, str]] = []
    beat_symbol_counts: Counter = Counter()

    for seg in segments:
        rec_path = os.path.join(dataset_dir, seg.record)
        # 读取标注文件（.atr）。extension="atr" 表示读取同名的 .atr 文件。
        ann = wfdb.rdann(rec_path, "atr")

        beat_symbol_counts.update(ann.symbol)

        # Beat 标注：ann.symbol 里除了 '+' 以外，通常都代表一个 beat 类别（N/S/V/Q...）。
        # '+' 在 Icentia11k 里多用于提示“这里有节律（rhythm）标注事件”，不是一个 beat 类型。
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol == "+":
                continue
            global_sample = int(sample) + seg.start_sample_global
            beat_times_by_symbol[symbol].append(global_sample / seg.fs)

        # 节律区间解析（rhythm regions）：
        # aux_note 里保存的是事件 token（类似括号语言）：
        # - '(N' / '(AFIB' / '(AFL' 开始一个区间
        # - ')' 结束区间
        # 我们用 stack（栈）来“配对”开始与结束，从而得到区间 [start, end]。
        stack: List[Tuple[str, int]] = []
        for sample, aux in zip(ann.sample, ann.aux_note):
            aux = _safe_aux(aux)
            if not aux:
                continue
            global_sample = int(sample) + seg.start_sample_global

            if aux.startswith("("):
                label = aux[1:]
                if label.endswith(")"):
                    # 有些记录里会出现 "(N)" 这种“同一时间点开始又结束”的 token；
                    # 这种更像是瞬时标记（不是一个持续区间），这里不把它当成主要区间。
                    label = label[:-1]
                    rhythm_regions.append(
                        (global_sample / seg.fs, global_sample / seg.fs, label)
                    )
                    continue

                # 防御性处理：如果已经有一个区间没闭合，又遇到新的开始 token，
                # 我们就把旧区间强行在当前点闭合（避免 stack 一直堆积导致结果错乱）。
                if stack:
                    prev_label, prev_start = stack.pop()
                    rhythm_regions.append(
                        (prev_start / seg.fs, global_sample / seg.fs, prev_label)
                    )
                stack.append((label, global_sample))
                continue

            if aux == ")":
                if stack:
                    label, start_sample = stack.pop()
                    rhythm_regions.append(
                        (start_sample / seg.fs, global_sample / seg.fs, label)
                    )

        # 如果到 segment 末尾仍然有未闭合的区间（理论上不该发生，但数据里可能会有），
        # 我们把它们闭合到该 segment 的结束位置。
        if stack:
            end_sample = seg.start_sample_global + seg.sig_len
            while stack:
                label, start_sample = stack.pop()
                rhythm_regions.append((start_sample / seg.fs, end_sample / seg.fs, label))

    return beat_times_by_symbol, rhythm_regions, beat_symbol_counts


def compute_envelope(
    dataset_dir: str, segments: Sequence[SegmentInfo], window_sec: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a global min/max envelope (in physical units) with a fixed window.

    Returns:
      t_sec: global time (seconds)
      y_min: min per window
      y_max: max per window
    """
    if not segments:
        raise ValueError("No segments given")

    fs = segments[0].fs
    # 把“窗口秒数”转成“窗口采样点数”。例如 fs=250Hz，window_sec=1s => 250 点/窗
    window_samples = int(round(window_sec * fs))
    if window_samples < 1:
        raise ValueError("window_sec too small")

    t_chunks: List[np.ndarray] = []
    min_chunks: List[np.ndarray] = []
    max_chunks: List[np.ndarray] = []

    for seg in segments:
        rec_path = os.path.join(dataset_dir, seg.record)
        # 只读取第 1 个通道（单导联 ECG），减少 IO 和内存。
        # wfdb.rdrecord 默认 physical=True，会把 int16 按 .hea 的增益换算成物理单位（mV），结果在 rec.p_signal。
        rec = wfdb.rdrecord(rec_path, channels=[0])
        if rec.p_signal is None:
            raise SystemExit(f"Missing p_signal for record: {seg.record}")
        signal = rec.p_signal[:, 0].astype(np.float32, copy=False)

        # 只取能整除窗口长度的部分，方便 reshape 成 (n_windows, window_samples)
        n_full = signal.shape[0] // window_samples
        if n_full <= 0:
            continue

        trimmed = signal[: n_full * window_samples]
        windows = trimmed.reshape(n_full, window_samples)
        # 每个窗口取最小/最大值，形成 envelope（包络）。
        y_min = windows.min(axis=1)
        y_max = windows.max(axis=1)

        # x 坐标用“窗口中心点时间”（更直观），并加上该 segment 在拼接序列中的起始时间偏移。
        centers = (np.arange(n_full, dtype=np.float32) * window_samples + window_samples / 2) / fs
        t_sec = centers + seg.start_sec_global

        t_chunks.append(t_sec)
        min_chunks.append(y_min)
        max_chunks.append(y_max)

    if not t_chunks:
        raise SystemExit("No envelope points computed (check dataset integrity).")

    return np.concatenate(t_chunks), np.concatenate(min_chunks), np.concatenate(max_chunks)


def pick_zoom_requests(
    segments: Sequence[SegmentInfo],
    beat_times_by_symbol: DefaultDict[str, List[float]],
    dataset_dir: str,
    max_windows: int = 2,
) -> List[ZoomRequest]:
    """
    Pick a few interesting windows to zoom in on.

    Priority: V (PVC) -> S (PAC) -> Q (unclassifiable) -> N (normal).
    """
    # 选择“值得放大看”的 beat：
    # - 在长程图里，很多细节看不出来，所以我们会挑一些少见/异常的 beat 作为示例窗口。
    # - 优先级：V（室早）> S（房早）> Q（无法分类）> N（正常）
    candidates: List[ZoomRequest] = []
    priority = ["V", "S", "Q", "N"]

    for symbol in priority:
        if symbol not in beat_times_by_symbol:
            continue
        if len(candidates) >= max_windows:
            break

        # Find the earliest occurrence in the concatenated timeline.
        t_global = min(beat_times_by_symbol[symbol])
        global_sample = int(round(t_global * segments[0].fs))

        # Locate which segment this global sample falls into.
        for seg in segments:
            if seg.start_sample_global <= global_sample < seg.start_sample_global + seg.sig_len:
                sample_in_record = global_sample - seg.start_sample_global
                candidates.append(
                    ZoomRequest(
                        title=f"Zoom around '{symbol}' beat ({seg.record}, sample {sample_in_record})",
                        record=seg.record,
                        sample_in_record=sample_in_record,
                        fs=seg.fs,
                        kind=symbol,
                    )
                )
                break

    # If we have fewer than max_windows but do have multiple rare beats (e.g., multiple S),
    # try to add a later example of the rarest available symbol.
    if len(candidates) < max_windows:
        for symbol in ["V", "S", "Q"]:
            times = beat_times_by_symbol.get(symbol, [])
            if len(times) < 2:
                continue
            times_sorted = sorted(times)
            # Pick a later example (middle) to show a different location.
            t_global = times_sorted[len(times_sorted) // 2]
            global_sample = int(round(t_global * segments[0].fs))
            for seg in segments:
                if seg.start_sample_global <= global_sample < seg.start_sample_global + seg.sig_len:
                    sample_in_record = global_sample - seg.start_sample_global
                    title = f"Zoom around another '{symbol}' beat ({seg.record}, sample {sample_in_record})"
                    # Avoid duplicates.
                    if any(z.record == seg.record and z.sample_in_record == sample_in_record for z in candidates):
                        break
                    candidates.append(
                        ZoomRequest(
                            title=title,
                            record=seg.record,
                            sample_in_record=sample_in_record,
                            fs=seg.fs,
                            kind=symbol,
                        )
                    )
                    break
            break

    return candidates[:max_windows]


def _unique_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if not l:
            continue
        uniq.setdefault(l, h)
    if uniq:
        ax.legend(list(uniq.values()), list(uniq.keys()), loc="upper right", frameon=True)


def plot_patient_overview(
    dataset_dir: str,
    segments: Sequence[SegmentInfo],
    out_path: str,
    window_sec: float,
    zoom_half_window_sec: float,
    dpi: int,
) -> str:
    # 1) 读取并解析标注（beat + rhythm 区间）
    # 2) 计算长程 envelope（避免直接画几十小时的原始采样点）
    beat_times_by_symbol, rhythm_regions, beat_symbol_counts = extract_annotations(
        dataset_dir, segments
    )
    t_sec, y_min, y_max = compute_envelope(dataset_dir, segments, window_sec=window_sec)

    # total_hours 是“拼接后的时长”，不一定等于真实连续佩戴时长（子集数据可能有缺口）。
    total_hours = (segments[-1].start_sample_global + segments[-1].sig_len) / segments[0].fs / 3600.0
    patient_id = segments[0].patient_id
    fs = segments[0].fs

    # Pick zoom windows.
    zooms = pick_zoom_requests(segments, beat_times_by_symbol, dataset_dir, max_windows=2)

    nrows = 2 + len(zooms)
    fig = plt.figure(figsize=(20, 4.2 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows, 1, height_ratios=[2.2, 0.9] + [1.6] * len(zooms))

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    # --- Panel 1: envelope + rhythm shading + segment boundaries ---
    # Panel 1：长程 ECG envelope（填充 min/max 区间）
    ax0.fill_between(
        t_sec / 3600.0,
        y_min,
        y_max,
        color="#2F4858",
        alpha=0.55,
        linewidth=0.0,
        label=f"ECG envelope (min/max per {window_sec:g}s)",
    )

    rhythm_colors = {
        "N": "#3A7CA5",  # NSR
        "AFIB": "#E76F51",
        "AFL": "#9B5DE5",
    }
    # 把节律区间画成背景色块（类似“状态高亮”）
    for start, end, label in rhythm_regions:
        if end <= start:
            continue
        color = rhythm_colors.get(label, "#BBBBBB")
        ax0.axvspan(
            start / 3600.0,
            end / 3600.0,
            color=color,
            alpha=0.12,
            linewidth=0,
            label=f"Rhythm: {label}",
        )

    # segment 边界虚线：提醒拼接点在哪里
    for seg in segments[1:]:
        ax0.axvline(seg.start_sec_global / 3600.0, color="black", linestyle="--", linewidth=0.7, alpha=0.45)

    ax0.set_title(
        f"Icentia11k patient p{patient_id:05d} | {len(segments)} segments | "
        f"{total_hours:.2f} hours total (concatenated) | fs={fs:g}Hz"
    )
    ax0.set_xlabel("Time (hours, concatenated segments)")
    ax0.set_ylabel("ECG (mV)")
    ax0.grid(True, alpha=0.2)
    _unique_legend(ax0)

    # --- Panel 2: beat marker timeline (focus on non-normal beats) ---
    # Show only beats that are informative at scale (Q/S/V). N is too dense.
    # Panel 2：beat 时间线（eventplot）。
    # 这里不画 N（正常）是因为它数量巨大，会让图变成一整条密集黑带；
    # 只画 V/S/Q 更能突出“异常/稀有事件”在长程中的位置分布。
    y_rows = []
    positions = []
    colors = []
    labels = []

    beat_color = {
        "Q": "#6C757D",  # gray
        "S": "#F4A261",  # orange
        "V": "#E63946",  # red
        "N": "#2A9D8F",  # green (usually omitted)
    }

    row_order = ["V", "S", "Q"]
    for row_idx, sym in enumerate(row_order):
        times = beat_times_by_symbol.get(sym, [])
        if not times:
            continue
        positions.append(np.asarray(times, dtype=np.float32) / 3600.0)
        y_rows.append(float(row_idx))
        colors.append(beat_color.get(sym, "black"))
        labels.append(f"Beat: {sym} (count={len(times)})")

    if positions:
        ax1.eventplot(
            positions,
            lineoffsets=y_rows,
            linelengths=0.85,
            colors=colors,
            linewidths=0.8,
        )

        ax1.set_yticks(y_rows)
        ax1.set_yticklabels([f"{sym}" for sym in row_order if beat_times_by_symbol.get(sym)])
    else:
        ax1.text(
            0.01,
            0.5,
            "No non-normal beat symbols found to display (only 'N').",
            transform=ax1.transAxes,
            ha="left",
            va="center",
        )

    for start, end, label in rhythm_regions:
        if end <= start:
            continue
        color = rhythm_colors.get(label, "#BBBBBB")
        ax1.axvspan(start / 3600.0, end / 3600.0, color=color, alpha=0.08, linewidth=0)

    for seg in segments[1:]:
        ax1.axvline(seg.start_sec_global / 3600.0, color="black", linestyle="--", linewidth=0.7, alpha=0.45)

    ax1.set_xlabel("Time (hours, concatenated segments)")
    ax1.set_title("Annotation timeline (non-normal beats + rhythm regions)")
    ax1.grid(True, alpha=0.2)
    if labels:
        # Custom legend for eventplot rows
        legend_handles = [
            plt.Line2D([0], [0], color=beat_color.get(sym, "black"), lw=2)
            for sym in row_order
            if beat_times_by_symbol.get(sym)
        ]
        legend_labels = [
            f"{sym} ({len(beat_times_by_symbol.get(sym, []))})"
            for sym in row_order
            if beat_times_by_symbol.get(sym)
        ]
        ax1.legend(legend_handles, legend_labels, loc="upper right", frameon=True, title="Beat counts")

    # --- Zoom panels ---
    # 局部放大图：在某个 beat 附近截取几秒钟，画“原始波形折线”，并把该窗口内的 beat 标注点散点叠加上去。
    for idx, zoom in enumerate(zooms):
        ax = fig.add_subplot(gs[2 + idx, 0])
        rec_path = os.path.join(dataset_dir, zoom.record)
        half = int(round(zoom_half_window_sec * zoom.fs))
        start = max(0, int(zoom.sample_in_record) - half)
        end = int(zoom.sample_in_record) + half
        # Clamp to record length
        header = wfdb.rdheader(rec_path)
        if header.sig_len is not None:
            end = min(end, int(header.sig_len))
        if end <= start + 1:
            continue

        # 只读取局部窗口（sampfrom/sampto），避免加载整段 70 分钟数据来画一个 8 秒小图。
        rec = wfdb.rdrecord(rec_path, sampfrom=start, sampto=end, channels=[0])
        signal = rec.p_signal[:, 0]
        t = (np.arange(signal.shape[0]) + start) / zoom.fs

        ax.plot(t, signal, color="#1D3557", linewidth=1.0, label="ECG (mV)")

        # 读取同一窗口内的标注。shift_samps=True 表示把 ann.sample 变成“相对窗口起点”的索引，
        # 这样可以直接用它去索引 signal 数组（0..窗口长度-1）。
        ann = wfdb.rdann(rec_path, "atr", sampfrom=start, sampto=end, shift_samps=True)
        for symbol in ["N", "Q", "S", "V"]:
            samples = [s for s, sym in zip(ann.sample, ann.symbol) if sym == symbol]
            if not samples:
                continue
            samples = np.asarray(samples, dtype=int)
            samples = samples[(samples >= 0) & (samples < signal.shape[0])]
            if samples.size == 0:
                continue
            ax.scatter(
                (samples + start) / zoom.fs,
                signal[samples],
                s=18,
                color=beat_color.get(symbol, "black"),
                label=f"{symbol}",
                zorder=3,
            )

        ax.axvline(zoom.sample_in_record / zoom.fs, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(zoom.title)
        ax.set_xlabel("Time (s, within segment)")
        ax.set_ylabel("ECG (mV)")
        ax.grid(True, alpha=0.25)
        _unique_legend(ax)

    # Footer style info.
    fig.suptitle(
        f"Beat symbols present: {', '.join(sorted(beat_symbol_counts.keys()))}",
        fontsize=11,
        y=1.01,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate Icentia11k segments for a patient and plot a long-range ECG + annotations."
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
        help="Output image path (.png). Default: ECG_Model/viz_output/icentia_pXXXXX_overview.png",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Envelope window size in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--zoom-half-sec",
        type=float,
        default=4.0,
        help="Half window size (seconds) for each zoom subplot (default: 4.0 => 8s total).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150).",
    )
    return parser.parse_args()


def main() -> None:
    # 保持输出更干净：某些环境里 wfdb + numpy 会触发 longdouble dtype 的警告，这里屏蔽掉。
    warnings.filterwarnings("ignore", category=UserWarning, message=".*broken support for the dtype.*")

    args = parse_args()
    # 兼容两种目录结构：
    # - 你直接把 WFDB 文件放在 ECG_Model/dataset 下
    # - 或者放在 ECG_Model/dataset/Icentia11k 下
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
        raise SystemExit(f"Patient p{patient_id:05d} not found. Available: {pids[:10]}{'...' if len(pids)>10 else ''}")

    segments = build_patient_segments(dataset_dir, records, patient_id=patient_id)
    out_path = args.out
    if out_path is None:
        outdir = os.path.join(os.path.dirname(__file__), "viz_output")
        out_path = os.path.join(outdir, f"icentia_p{patient_id:05d}_overview.png")

    saved = plot_patient_overview(
        dataset_dir=dataset_dir,
        segments=segments,
        out_path=os.path.abspath(out_path),
        window_sec=float(args.window_sec),
        zoom_half_window_sec=float(args.zoom_half_sec),
        dpi=int(args.dpi),
    )
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
