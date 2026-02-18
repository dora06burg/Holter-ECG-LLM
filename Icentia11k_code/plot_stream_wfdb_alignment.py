#!/usr/bin/env python3
"""
可视化（对齐检查）：LSL 切窗后 WFDB 数据集
==========================================

你给的目录：
  ECG_Model/stream_result/lsl_wfdb_p00000_s00_s49_speed500

里面是一堆被“切窗/分割”后的 WFDB records（.hea/.dat/.atr），文件名形如：
  stream_p00000_000000.hea/.dat/.atr

这个脚本的目标不是画“整段长程拼接”（那样 beat 标注点太密看不清），而是专门帮你检查：
  标注点（ann.sample）是否和心跳（通常是 R 峰附近）对齐一致。

做法：
1) 遍历 records，读取信号 + 标注（忽略 symbol == '+' 的节律事件）。
2) 对每个 beat 标注点 s，去 s±search_ms 的小窗口里找一个“局部峰值点”（默认取 |x| 最大）。
   - offset = peak_sample - ann_sample
   - 如果标注对齐正确，offset 应该集中在 0ms 附近（可能有少量 ±1~2 个采样点抖动）。
3) 输出一张总览图：
   - offset 直方图
   - 每个 record 的 offset 统计（median / p95(|offset|) / max(|offset|)）
   - 若干个“最差 offset”示例的局部放大图（直接肉眼看标注线是否落在 R 峰上）

运行示例：
  python ECG_Model/plot_stream_wfdb_alignment.py \\
    --dataset ECG_Model/stream_result/lsl_wfdb_p00000_s00_s49_speed500 \\
    --out ECG_Model/viz_output/lsl_wfdb_alignment_overview.png

只检查某一条 record：
  python ECG_Model/plot_stream_wfdb_alignment.py \\
    --dataset ECG_Model/stream_result/lsl_wfdb_p00000_s00_s49_speed500 \\
    --record stream_p00000_000050 \\
    --out ECG_Model/viz_output/record_000050_alignment.png

依赖：
  pip install wfdb matplotlib numpy
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# 某些环境里 numpy 在探测 longdouble 时会发出告警（信息量不大但很吵）。
# 需要在 import numpy 之前设置过滤器，才能保证干净输出。
# 这个 warning 的 message 里带换行，re 默认 '.' 不匹配换行，所以这里用 DOTALL 模式。
warnings.filterwarnings("ignore", category=UserWarning, message=r"(?s).*broken support for the dtype.*")

import numpy as np

try:
    import wfdb
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency: wfdb. Install with `pip install wfdb`.") from exc


RE_STREAM_BASENAME = re.compile(r"^stream_p(?P<pid>\d{5})_(?P<chunk>\d+)$")


@dataclass(frozen=True)
class RecordInfo:
    record: str  # record base name (relative to dataset_dir, no extension)
    record_index: int  # index in sorted record list
    fs: float
    sig_len: int
    start_sample_global: int
    patient_id: Optional[int]
    chunk_id: Optional[int]

    @property
    def start_time_hours(self) -> float:
        return float(self.start_sample_global) / float(self.fs) / 3600.0


@dataclass(frozen=True)
class BeatExample:
    record: str
    fs: float
    sample_in_record: int
    symbol: str
    peak_sample_in_record: int
    offset_samples: int

    @property
    def offset_ms(self) -> float:
        return float(self.offset_samples) * 1000.0 / float(self.fs)


def _parse_comments_to_dict(comments: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in comments:
        if "=" not in c:
            continue
        k, v = c.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def read_records_list(dataset_dir: str) -> List[str]:
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


def _parse_stream_name(record: str) -> Tuple[Optional[int], Optional[int]]:
    base = os.path.basename(record)
    m = RE_STREAM_BASENAME.match(base)
    if not m:
        return None, None
    return int(m.group("pid")), int(m.group("chunk"))


def build_record_infos(dataset_dir: str, records: Sequence[str]) -> List[RecordInfo]:
    infos: List[RecordInfo] = []
    fs_ref: Optional[float] = None

    for idx, rec in enumerate(records):
        header = wfdb.rdheader(os.path.join(dataset_dir, rec))
        if header.fs is None or header.sig_len is None:
            raise SystemExit(f"Missing fs/sig_len in header for record: {rec}")

        fs = float(header.fs)
        sig_len = int(header.sig_len)
        if fs_ref is None:
            fs_ref = fs
        elif abs(fs_ref - fs) > 1e-9:
            raise SystemExit(f"Inconsistent fs across records: {fs_ref} vs {fs} (record={rec})")

        meta = _parse_comments_to_dict(getattr(header, "comments", []) or [])
        start_sample = meta.get("global_start_sample")
        start_sample_global = int(start_sample) if start_sample is not None else 0

        pid, chunk = _parse_stream_name(rec)
        infos.append(
            RecordInfo(
                record=rec,
                record_index=idx,
                fs=fs,
                sig_len=sig_len,
                start_sample_global=start_sample_global,
                patient_id=pid,
                chunk_id=chunk,
            )
        )

    # Prefer sorting by global_start_sample if it looks sane (strictly increasing for most rows).
    if infos and all(i.start_sample_global is not None for i in infos):
        infos_sorted = sorted(infos, key=lambda x: (x.start_sample_global, x.record))
        return [RecordInfo(**{**i.__dict__, "record_index": j}) for j, i in enumerate(infos_sorted)]

    return infos


def _extract_beat_samples_and_symbols(ann) -> Tuple[np.ndarray, List[str]]:
    samples: List[int] = []
    symbols: List[str] = []
    for s, sym in zip(ann.sample.tolist(), ann.symbol):
        if sym == "+":  # rhythm token, not a beat
            continue
        samples.append(int(s))
        symbols.append(str(sym))
    return np.asarray(samples, dtype=np.int64), symbols


def _find_local_peak_sample(
    sig: np.ndarray,
    center_sample: int,
    search_radius: int,
    *,
    peak_mode: str,
) -> int:
    if search_radius <= 0:
        return int(center_sample)
    start = max(0, int(center_sample) - int(search_radius))
    end = min(int(sig.shape[0]), int(center_sample) + int(search_radius) + 1)
    if end <= start:
        return int(center_sample)
    w = sig[start:end]
    if peak_mode == "abs":
        idx = int(np.argmax(np.abs(w)))
    elif peak_mode == "max":
        idx = int(np.argmax(w))
    elif peak_mode == "min":
        idx = int(np.argmin(w))
    else:
        raise ValueError(f"Unknown peak_mode: {peak_mode}")
    return int(start + idx)


@dataclass(frozen=True)
class RecordOffsetStats:
    record: str
    start_time_hours: float
    n_beats: int
    median_offset_ms: float
    p95_abs_offset_ms: float
    max_abs_offset_ms: float


@dataclass(frozen=True)
class AlignmentAnalysis:
    fs: float
    beat_record: List[str]
    beat_sample: np.ndarray  # (n,)
    beat_symbol: List[str]
    beat_peak_sample: np.ndarray  # (n,)
    beat_offset_samples: np.ndarray  # (n,)
    per_record_stats: List[RecordOffsetStats]


def analyze_alignment(
    dataset_dir: str,
    infos: Sequence[RecordInfo],
    *,
    search_ms: float,
    peak_mode: str,
    max_beats_per_record: int,
) -> AlignmentAnalysis:
    if not infos:
        raise ValueError("No records to analyze")

    fs = float(infos[0].fs)
    search_radius = int(round(float(search_ms) * fs / 1000.0))

    beat_record: List[str] = []
    beat_sample: List[int] = []
    beat_symbol: List[str] = []
    beat_peak: List[int] = []
    beat_offset: List[int] = []
    per_record_stats: List[RecordOffsetStats] = []

    for info in infos:
        rec_path = os.path.join(dataset_dir, info.record)

        rec = wfdb.rdrecord(rec_path, channels=[0])
        if rec.p_signal is not None:
            sig = rec.p_signal[:, 0].astype(np.float32, copy=False)
        elif rec.d_signal is not None:
            sig = rec.d_signal[:, 0].astype(np.float32, copy=False)
        else:
            continue

        ann_path = f"{rec_path}.atr"
        if not os.path.isfile(ann_path):
            per_record_stats.append(
                RecordOffsetStats(
                    record=info.record,
                    start_time_hours=info.start_time_hours,
                    n_beats=0,
                    median_offset_ms=float("nan"),
                    p95_abs_offset_ms=float("nan"),
                    max_abs_offset_ms=float("nan"),
                )
            )
            continue

        try:
            ann = wfdb.rdann(rec_path, "atr")
        except Exception:
            per_record_stats.append(
                RecordOffsetStats(
                    record=info.record,
                    start_time_hours=info.start_time_hours,
                    n_beats=0,
                    median_offset_ms=float("nan"),
                    p95_abs_offset_ms=float("nan"),
                    max_abs_offset_ms=float("nan"),
                )
            )
            continue

        samples, symbols = _extract_beat_samples_and_symbols(ann)
        if samples.size == 0:
            per_record_stats.append(
                RecordOffsetStats(
                    record=info.record,
                    start_time_hours=info.start_time_hours,
                    n_beats=0,
                    median_offset_ms=float("nan"),
                    p95_abs_offset_ms=float("nan"),
                    max_abs_offset_ms=float("nan"),
                )
            )
            continue

        # Clamp and optionally subsample beats (avoid pathological huge annotation lists).
        valid = (samples >= 0) & (samples < sig.shape[0])
        samples = samples[valid]
        symbols = [sym for sym, ok in zip(symbols, valid.tolist()) if ok]

        if max_beats_per_record > 0 and samples.size > max_beats_per_record:
            step = int(np.ceil(samples.size / float(max_beats_per_record)))
            samples = samples[::step]
            symbols = symbols[::step]

        offsets_this: List[float] = []
        abs_offsets_this: List[float] = []
        for s, sym in zip(samples.tolist(), symbols):
            peak_s = _find_local_peak_sample(sig, int(s), search_radius, peak_mode=peak_mode)
            off = int(peak_s) - int(s)
            beat_record.append(info.record)
            beat_sample.append(int(s))
            beat_symbol.append(sym)
            beat_peak.append(int(peak_s))
            beat_offset.append(int(off))
            offsets_this.append(float(off) * 1000.0 / fs)
            abs_offsets_this.append(abs(float(off)) * 1000.0 / fs)

        offsets_arr = np.asarray(offsets_this, dtype=np.float32)
        abs_offsets_arr = np.asarray(abs_offsets_this, dtype=np.float32)
        per_record_stats.append(
            RecordOffsetStats(
                record=info.record,
                start_time_hours=info.start_time_hours,
                n_beats=int(offsets_arr.size),
                median_offset_ms=float(np.median(offsets_arr)) if offsets_arr.size else float("nan"),
                p95_abs_offset_ms=float(np.percentile(abs_offsets_arr, 95)) if abs_offsets_arr.size else float("nan"),
                max_abs_offset_ms=float(np.max(abs_offsets_arr)) if abs_offsets_arr.size else float("nan"),
            )
        )

    return AlignmentAnalysis(
        fs=fs,
        beat_record=beat_record,
        beat_sample=np.asarray(beat_sample, dtype=np.int64),
        beat_symbol=beat_symbol,
        beat_peak_sample=np.asarray(beat_peak, dtype=np.int64),
        beat_offset_samples=np.asarray(beat_offset, dtype=np.int64),
        per_record_stats=per_record_stats,
    )


def select_examples(
    analysis: AlignmentAnalysis,
    *,
    num_worst: int,
    num_random: int,
    seed: int,
) -> List[BeatExample]:
    n = int(analysis.beat_offset_samples.size)
    if n <= 0:
        return []

    abs_ms = np.abs(analysis.beat_offset_samples.astype(np.float32)) * 1000.0 / float(analysis.fs)
    k = min(int(num_worst), n)
    if k > 0:
        idx_worst = np.argpartition(abs_ms, -k)[-k:]
        idx_worst = idx_worst[np.argsort(abs_ms[idx_worst])[::-1]]
    else:
        idx_worst = np.asarray([], dtype=np.int64)

    examples: List[BeatExample] = []
    for i in idx_worst.tolist():
        examples.append(
            BeatExample(
                record=analysis.beat_record[i],
                fs=float(analysis.fs),
                sample_in_record=int(analysis.beat_sample[i]),
                symbol=str(analysis.beat_symbol[i]),
                peak_sample_in_record=int(analysis.beat_peak_sample[i]),
                offset_samples=int(analysis.beat_offset_samples[i]),
            )
        )

    if int(num_random) > 0:
        rng = np.random.default_rng(int(seed))
        exclude = set(idx_worst.tolist())
        pool = np.asarray([i for i in range(n) if i not in exclude], dtype=np.int64)
        if pool.size > 0:
            r = min(int(num_random), int(pool.size))
            idx_rand = rng.choice(pool, size=r, replace=False)
            for i in idx_rand.tolist():
                examples.append(
                    BeatExample(
                        record=analysis.beat_record[i],
                        fs=float(analysis.fs),
                        sample_in_record=int(analysis.beat_sample[i]),
                        symbol=str(analysis.beat_symbol[i]),
                        peak_sample_in_record=int(analysis.beat_peak_sample[i]),
                        offset_samples=int(analysis.beat_offset_samples[i]),
                    )
                )
    return examples


def write_record_stats_csv(path: str, stats: Sequence[RecordOffsetStats]) -> str:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["record", "start_time_hours", "n_beats", "median_offset_ms", "p95_abs_offset_ms", "max_abs_offset_ms"])
        for s in stats:
            w.writerow(
                [
                    s.record,
                    f"{s.start_time_hours:.6f}",
                    int(s.n_beats),
                    f"{s.median_offset_ms:.6f}" if np.isfinite(s.median_offset_ms) else "",
                    f"{s.p95_abs_offset_ms:.6f}" if np.isfinite(s.p95_abs_offset_ms) else "",
                    f"{s.max_abs_offset_ms:.6f}" if np.isfinite(s.max_abs_offset_ms) else "",
                ]
            )
    return path


def _plot_zoom(
    ax,
    *,
    dataset_dir: str,
    ex: BeatExample,
    half_window_sec: float,
    search_ms: float,
    peak_mode: str,
) -> None:
    rec_path = os.path.join(dataset_dir, ex.record)
    half = int(round(float(half_window_sec) * float(ex.fs)))
    start = max(0, int(ex.sample_in_record) - half)
    end = int(ex.sample_in_record) + half + 1

    # Clamp to record length
    header = wfdb.rdheader(rec_path)
    if header.sig_len is not None:
        end = min(end, int(header.sig_len))
    if end <= start + 1:
        ax.set_axis_off()
        return

    rec = wfdb.rdrecord(rec_path, sampfrom=start, sampto=end, channels=[0])
    if rec.p_signal is not None:
        sig = rec.p_signal[:, 0].astype(np.float32, copy=False)
    elif rec.d_signal is not None:
        sig = rec.d_signal[:, 0].astype(np.float32, copy=False)
    else:
        ax.set_axis_off()
        return

    t = (np.arange(sig.shape[0], dtype=np.float32) + start) / float(ex.fs)
    ax.plot(t, sig, color="#1D3557", linewidth=1.0, label="ECG (mV)")

    # Overlay annotations within this window
    try:
        ann = wfdb.rdann(rec_path, "atr", sampfrom=start, sampto=end, shift_samps=True)
    except Exception:
        ann = None

    beat_colors = {"N": "#1f77b4", "S": "#ff7f0e", "V": "#d62728", "Q": "#9467bd"}
    if ann is not None and len(ann.sample) > 0:
        for sym in ["N", "Q", "S", "V"]:
            samples = [s for s, s_sym in zip(ann.sample.tolist(), ann.symbol) if s_sym == sym]
            if not samples:
                continue
            samples = np.asarray(samples, dtype=np.int64)
            samples = samples[(samples >= 0) & (samples < sig.shape[0])]
            if samples.size == 0:
                continue
            ax.scatter(
                (samples + start) / float(ex.fs),
                sig[samples],
                s=18,
                color=beat_colors.get(sym, "#666666"),
                label=sym,
                zorder=3,
                alpha=0.9,
                edgecolors="none",
            )

    # Highlight the annotation sample and the detected local peak sample
    ax.axvline(ex.sample_in_record / float(ex.fs), color="#2B2D42", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(ex.peak_sample_in_record / float(ex.fs), color="#E63946", linestyle=":", linewidth=1.2, alpha=0.9)

    ax.set_title(
        f"{ex.record} | sym={ex.symbol} | offset={ex.offset_ms:+.2f}ms | search={search_ms:g}ms ({peak_mode})"
    )
    ax.set_xlabel("Time (s, within record)")
    ax.set_ylabel("ECG (mV)")
    ax.grid(True, alpha=0.25)


def plot_alignment_overview(
    *,
    dataset_dir: str,
    infos: Sequence[RecordInfo],
    analysis: AlignmentAnalysis,
    examples: Sequence[BeatExample],
    out_path: str,
    half_window_sec: float,
    search_ms: float,
    peak_mode: str,
    dpi: int,
) -> str:
    # Headless-safe backend; user can open the PNG for inspection.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    offsets_ms = analysis.beat_offset_samples.astype(np.float32) * 1000.0 / float(analysis.fs)
    offsets_ms = offsets_ms[np.isfinite(offsets_ms)]

    zooms = list(examples)
    n_zooms = len(zooms)
    ncols = 2
    zoom_rows = int(np.ceil(max(1, n_zooms) / float(ncols)))

    fig_h = 4.8 + 3.2 * zoom_rows
    fig = plt.figure(figsize=(14, fig_h), dpi=int(dpi))
    gs = fig.add_gridspec(nrows=2 + zoom_rows, ncols=2, height_ratios=[1.1, 1.0] + [1.2] * zoom_rows)

    # --- Histogram panel ---
    ax_hist = fig.add_subplot(gs[0, :])
    if offsets_ms.size:
        # Adaptive range but keep it sane for readability.
        p99 = float(np.percentile(np.abs(offsets_ms), 99))
        lim = max(5.0, min(80.0, p99 * 1.2))
        bins = np.linspace(-lim, lim, 81)
        ax_hist.hist(offsets_ms, bins=bins, color="#3A7CA5", edgecolor="white", alpha=0.95)
        ax_hist.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax_hist.set_xlim(-lim, lim)
        ax_hist.set_title(
            f"Annotation alignment offsets (ms) | beats={offsets_ms.size:,} | fs={analysis.fs:g}Hz"
        )
        ax_hist.set_xlabel("offset = (local_peak_sample - ann.sample) [ms]")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, alpha=0.2)
        med = float(np.median(offsets_ms))
        p95 = float(np.percentile(np.abs(offsets_ms), 95))
        ax_hist.text(
            0.99,
            0.97,
            f"median={med:+.2f}ms\np95(|offset|)={p95:.2f}ms\nsearch={search_ms:g}ms ({peak_mode})",
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#DDDDDD"},
        )
    else:
        ax_hist.text(0.5, 0.5, "No beat annotations found.", transform=ax_hist.transAxes, ha="center")
        ax_hist.set_axis_off()

    # --- Per-record stats panel ---
    ax_stats = fig.add_subplot(gs[1, :])
    stats = list(analysis.per_record_stats)
    xs = np.asarray([s.start_time_hours for s in stats], dtype=np.float32)
    med = np.asarray([s.median_offset_ms for s in stats], dtype=np.float32)
    p95 = np.asarray([s.p95_abs_offset_ms for s in stats], dtype=np.float32)
    mmax = np.asarray([s.max_abs_offset_ms for s in stats], dtype=np.float32)

    ax_stats.plot(xs, med, color="#1D3557", linewidth=1.2, label="median(offset) [ms]")
    ax_stats.plot(xs, p95, color="#E76F51", linewidth=1.2, label="p95(|offset|) [ms]")
    ax_stats.plot(xs, mmax, color="#9B2226", linewidth=1.0, alpha=0.75, label="max(|offset|) [ms]")
    ax_stats.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_stats.set_title("Per-record offset statistics (vs record start time)")
    ax_stats.set_xlabel("Record start time (hours, from global_start_sample)")
    ax_stats.set_ylabel("ms")
    ax_stats.grid(True, alpha=0.25)
    ax_stats.legend(loc="upper right", frameon=True)

    # --- Zoom panels ---
    if not zooms:
        ax = fig.add_subplot(gs[2, :])
        ax.text(0.5, 0.5, "No zoom examples selected.", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
    else:
        for k, ex in enumerate(zooms):
            r = k // ncols
            c = k % ncols
            ax = fig.add_subplot(gs[2 + r, c])
            _plot_zoom(
                ax,
                dataset_dir=dataset_dir,
                ex=ex,
                half_window_sec=float(half_window_sec),
                search_ms=float(search_ms),
                peak_mode=str(peak_mode),
            )

    # Overall title
    ds_name = os.path.basename(os.path.abspath(dataset_dir))
    fig.suptitle(f"WFDB stream windows alignment check | dataset={ds_name} | records={len(infos)}", y=0.995)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize alignment between WFDB beat annotations and ECG peaks.")
    p.add_argument("--dataset", type=str, required=True, help="Folder containing WFDB files + RECORDS.")
    p.add_argument("--out", type=str, default=None, help="Output PNG path.")
    p.add_argument(
        "--record",
        type=str,
        default=None,
        help="Only analyze a single record (e.g., stream_p00000_000050).",
    )
    p.add_argument(
        "--record-index",
        type=int,
        default=None,
        help="Only analyze one record by index in RECORDS (0-based).",
    )
    p.add_argument("--max-records", type=int, default=0, help="Analyze at most N records (0 = all).")
    p.add_argument("--search-ms", type=float, default=80.0, help="Search radius around ann.sample (ms).")
    p.add_argument(
        "--peak-mode",
        type=str,
        default="abs",
        choices=["abs", "max", "min"],
        help="How to pick the 'local peak' inside the search window.",
    )
    p.add_argument("--half-window-sec", type=float, default=1.5, help="Half window (sec) for each zoom plot.")
    p.add_argument("--num-worst", type=int, default=8, help="Number of worst-offset beats to zoom in on.")
    p.add_argument("--num-random", type=int, default=4, help="Also add N random beat zooms (helps sanity check).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-beats-per-record",
        type=int,
        default=0,
        help="Optionally cap beats processed per record (0 = no cap).",
    )
    p.add_argument(
        "--stats-csv",
        type=str,
        default=None,
        help="Optional CSV path to write per-record offset stats.",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset)
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"--dataset not found: {dataset_dir}")

    records = read_records_list(dataset_dir)
    if not records:
        raise SystemExit(f"No WFDB records found under: {dataset_dir}")

    # Narrow down to a single record if requested.
    if args.record is not None:
        if args.record not in records and not os.path.isfile(os.path.join(dataset_dir, f"{args.record}.hea")):
            raise SystemExit(f"--record not found in dataset: {args.record}")
        records = [args.record]
    elif args.record_index is not None:
        idx = int(args.record_index)
        if idx < 0 or idx >= len(records):
            raise SystemExit(f"--record-index out of range: {idx} (records={len(records)})")
        records = [records[idx]]
    elif args.max_records and int(args.max_records) > 0:
        records = records[: int(args.max_records)]

    infos = build_record_infos(dataset_dir, records)
    analysis = analyze_alignment(
        dataset_dir,
        infos,
        search_ms=float(args.search_ms),
        peak_mode=str(args.peak_mode),
        max_beats_per_record=int(args.max_beats_per_record),
    )

    examples = select_examples(
        analysis,
        num_worst=int(args.num_worst),
        num_random=int(args.num_random),
        seed=int(args.seed),
    )

    out_path = args.out
    if out_path is None:
        outdir = os.path.join(os.path.dirname(__file__), "viz_output")
        ds_name = os.path.basename(os.path.abspath(dataset_dir))
        if args.record is not None:
            out_path = os.path.join(outdir, f"{ds_name}_{args.record}_alignment.png")
        else:
            out_path = os.path.join(outdir, f"{ds_name}_alignment_overview.png")

    saved = plot_alignment_overview(
        dataset_dir=dataset_dir,
        infos=infos,
        analysis=analysis,
        examples=examples,
        out_path=os.path.abspath(out_path),
        half_window_sec=float(args.half_window_sec),
        search_ms=float(args.search_ms),
        peak_mode=str(args.peak_mode),
        dpi=int(args.dpi),
    )

    if args.stats_csv is not None:
        csv_path = write_record_stats_csv(os.path.abspath(args.stats_csv), analysis.per_record_stats)
        print(f"Wrote stats CSV: {csv_path}")

    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
