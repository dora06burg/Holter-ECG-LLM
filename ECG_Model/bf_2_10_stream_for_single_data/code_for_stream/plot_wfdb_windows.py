from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import wfdb

from icentia_wfdb import normalize_rhythm_label, safe_aux_note


@dataclass(frozen=True)
class RhythmRegion:
    start_samp: int
    end_samp: int
    label_raw: str

    @property
    def label_norm(self) -> str:
        return normalize_rhythm_label(self.label_raw)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot WFDB windows written by receiver_lsl_to_wfdb.py")
    p.add_argument("--wfdb-dir", type=str, required=True, help="Directory containing WFDB records and RECORDS.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for PNGs.")
    p.add_argument("--max-records", type=int, default=0, help="Plot at most N records (0 = all).")
    p.add_argument("--summary-n", type=int, default=0, help="Also generate a multi-panel summary.png of the first N records (0 = disabled).")
    p.add_argument("--dpi", type=int, default=150)
    return p


def _list_records(wfdb_dir: Path) -> List[str]:
    records_path = wfdb_dir / "RECORDS"
    if records_path.is_file():
        return [line.strip() for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Fallback: infer from .hea files
    recs = sorted({p.stem for p in wfdb_dir.glob("*.hea")})
    return recs


def _parse_comments_to_dict(comments: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in comments:
        if "=" not in c:
            continue
        k, v = c.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _extract_rhythm_regions(ann, *, sig_len: int) -> List[RhythmRegion]:
    regions: List[RhythmRegion] = []
    stack: List[Tuple[str, int]] = []
    for sample, symbol, aux in zip(ann.sample.tolist(), ann.symbol, ann.aux_note):
        if symbol != "+":
            continue
        aux = safe_aux_note(aux)
        if not aux:
            continue
        sample = int(sample)

        if aux.startswith("("):
            label = aux[1:]
            if label.endswith(")"):
                continue
            if stack:
                prev_label, prev_start = stack.pop()
                regions.append(RhythmRegion(prev_start, sample, prev_label))
            stack.append((label, sample))
            continue

        if aux == ")":
            if stack:
                label, start = stack.pop()
                regions.append(RhythmRegion(start, sample, label))

    while stack:
        label, start = stack.pop()
        regions.append(RhythmRegion(start, int(sig_len), label))
    regions.sort(key=lambda r: (r.start_samp, r.end_samp))
    return regions


def _plot_one_record(*, wfdb_dir: Path, record: str, out_path: Path, dpi: int) -> None:
    rec = wfdb.rdrecord(str(wfdb_dir / record), channels=[0])
    sig = rec.p_signal[:, 0].astype(np.float32, copy=False) if rec.p_signal is not None else rec.d_signal[:, 0].astype(np.float32)
    fs = float(rec.fs)
    sig_len = int(rec.sig_len)
    t = np.arange(sig_len, dtype=np.float32) / fs

    ann = None
    ann_path = wfdb_dir / f"{record}.atr"
    if ann_path.exists():
        ann = wfdb.rdann(str(wfdb_dir / record), "atr")

    meta = _parse_comments_to_dict(getattr(rec, "comments", []) or [])
    global_start = meta.get("global_start_sample", "?")
    source_record = meta.get("source_record_at_start", "")
    pid = meta.get("patient_id", "")

    # Prepare markers
    beat_samples: List[int] = []
    beat_symbols: List[str] = []
    rhythm_regions: List[RhythmRegion] = []
    if ann is not None:
        for s, sym in zip(ann.sample.tolist(), ann.symbol):
            if sym == "+":
                continue
            beat_samples.append(int(s))
            beat_symbols.append(str(sym))
        rhythm_regions = _extract_rhythm_regions(ann, sig_len=sig_len)

    beat_colors = {"N": "#1f77b4", "S": "#ff7f0e", "V": "#d62728", "Q": "#9467bd"}
    rhythm_colors = {"NSR": "#E9F7EF", "AFIB": "#FDEDEC", "AFL": "#FCF3CF"}

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)

    # Rhythm shading
    for region in rhythm_regions:
        label = region.label_norm
        color = rhythm_colors.get(label, "#EEEEEE")
        ax.axvspan(region.start_samp / fs, region.end_samp / fs, color=color, alpha=0.6, lw=0)

    ax.plot(t, sig, lw=1, color="#222222")

    # Beat scatter
    if beat_samples:
        bx = np.asarray(beat_samples, dtype=np.int64) / fs
        by = sig[np.clip(np.asarray(beat_samples, dtype=np.int64), 0, sig_len - 1)]
        colors = [beat_colors.get(sym, "#666666") for sym in beat_symbols]
        ax.scatter(bx, by, s=18, c=colors, marker="o", edgecolors="none", alpha=0.9, label="beats")

    title = f"{record} | fs={fs:g}Hz | global_start={global_start}"
    if pid:
        title = f"{pid} | " + title
    if source_record:
        title += f" | source_record={source_record}"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG (mV)")
    ax.grid(True, alpha=0.2)

    # Legend-like note
    if rhythm_regions:
        labels = sorted({r.label_norm for r in rhythm_regions})
        ax.text(0.01, 0.02, f"rhythm: {', '.join(labels)}", transform=ax.transAxes, ha="left", va="bottom")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_summary(*, wfdb_dir: Path, records: Sequence[str], out_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n = len(records)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False, dpi=dpi)
    if n == 1:
        axes = [axes]

    for ax, record in zip(axes, records):
        rec = wfdb.rdrecord(str(wfdb_dir / record), channels=[0])
        sig = (
            rec.p_signal[:, 0].astype(np.float32, copy=False)
            if rec.p_signal is not None
            else rec.d_signal[:, 0].astype(np.float32)
        )
        fs = float(rec.fs)
        sig_len = int(rec.sig_len)
        t = np.arange(sig_len, dtype=np.float32) / fs

        ann_path = wfdb_dir / f"{record}.atr"
        ann = wfdb.rdann(str(wfdb_dir / record), "atr") if ann_path.exists() else None

        beat_samples: List[int] = []
        beat_symbols: List[str] = []
        rhythm_regions: List[RhythmRegion] = []
        if ann is not None:
            for s, sym in zip(ann.sample.tolist(), ann.symbol):
                if sym == "+":
                    continue
                beat_samples.append(int(s))
                beat_symbols.append(str(sym))
            rhythm_regions = _extract_rhythm_regions(ann, sig_len=sig_len)

        beat_colors = {"N": "#1f77b4", "S": "#ff7f0e", "V": "#d62728", "Q": "#9467bd"}
        rhythm_colors = {"NSR": "#E9F7EF", "AFIB": "#FDEDEC", "AFL": "#FCF3CF"}

        for region in rhythm_regions:
            label = region.label_norm
            color = rhythm_colors.get(label, "#EEEEEE")
            ax.axvspan(region.start_samp / fs, region.end_samp / fs, color=color, alpha=0.6, lw=0)
        ax.plot(t, sig, lw=1, color="#222222")

        if beat_samples:
            bx = np.asarray(beat_samples, dtype=np.int64) / fs
            by = sig[np.clip(np.asarray(beat_samples, dtype=np.int64), 0, sig_len - 1)]
            colors = [beat_colors.get(sym, "#666666") for sym in beat_symbols]
            ax.scatter(bx, by, s=16, c=colors, marker="o", edgecolors="none", alpha=0.9)

        ax.set_title(record)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("mV")
        ax.grid(True, alpha=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = _build_argparser().parse_args()
    wfdb_dir = Path(args.wfdb_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _list_records(wfdb_dir)
    if args.max_records and int(args.max_records) > 0:
        records = records[: int(args.max_records)]

    for rec in records:
        _plot_one_record(wfdb_dir=wfdb_dir, record=rec, out_path=out_dir / f"{rec}.png", dpi=int(args.dpi))

    if args.summary_n and int(args.summary_n) > 0:
        n = min(int(args.summary_n), len(records))
        if n > 0:
            _plot_summary(wfdb_dir=wfdb_dir, records=records[:n], out_path=out_dir / "summary.png", dpi=int(args.dpi))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

