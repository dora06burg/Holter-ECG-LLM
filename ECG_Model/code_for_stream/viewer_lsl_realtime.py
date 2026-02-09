from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from icentia_wfdb import normalize_rhythm_label, safe_aux_note
from lsl_utils import import_pylsl


@dataclass
class BeatMarker:
    global_sample: int
    symbol: str


@dataclass
class RhythmToken:
    global_sample: int
    aux_note: str


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Realtime viewer for Icentia LSL ECG + markers.")
    p.add_argument("--lsl-name-ecg", type=str, default="icentia_ecg")
    p.add_argument("--lsl-name-ann", type=str, default="icentia_ann")
    p.add_argument("--resolve-timeout", type=float, default=10.0)

    p.add_argument("--display-sec", type=float, default=12.0, help="How many seconds to display in the rolling window.")
    p.add_argument("--refresh-hz", type=float, default=10.0, help="Plot refresh rate.")
    p.add_argument("--fs", type=float, default=0.0, help="Override sampling rate (0 = read from stream info).")
    p.add_argument("--duration-sec", type=float, default=0.0, help="Run for N seconds then exit (0 = run until Ctrl-C).")
    p.add_argument("--no-plot", action="store_true", help="Disable matplotlib GUI; print streaming stats to stdout.")
    return p


def _resolve_one_stream(pylsl, *, name: str, timeout: float):
    results = pylsl.resolve_byprop("name", name, timeout=timeout)
    if not results:
        raise SystemExit(f"Could not resolve LSL stream with name='{name}' (timeout={timeout}s)")
    return results[0]


def _parse_marker(sample: List[str]) -> Optional[Dict[str, Any]]:
    if not sample:
        return None
    try:
        raw = json.loads(sample[0])
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def main() -> int:
    args = _build_argparser().parse_args()
    pylsl = import_pylsl()

    stream_ecg = _resolve_one_stream(pylsl, name=str(args.lsl_name_ecg), timeout=float(args.resolve_timeout))
    stream_ann = _resolve_one_stream(pylsl, name=str(args.lsl_name_ann), timeout=float(args.resolve_timeout))
    inlet_ecg = pylsl.StreamInlet(stream_ecg, max_buflen=60)
    inlet_ann = pylsl.StreamInlet(stream_ann, max_buflen=60)

    fs = float(args.fs) if float(args.fs) > 0 else float(inlet_ecg.info().nominal_srate())
    if fs <= 0:
        raise SystemExit("ECG stream nominal_srate is not set. Use --fs to override.")

    display_samples = int(round(float(args.display_sec) * fs))
    display_samples = max(10, display_samples)

    ecg: Deque[float] = deque(maxlen=display_samples)
    beats: List[BeatMarker] = []
    rhythm_tokens: List[RhythmToken] = []

    global_cursor = 0
    rhythm_stack: List[str] = []
    last_segment: Optional[str] = None

    # Plot (optional)
    fig = ax = line = beat_scatter = txt = None
    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))
        (line,) = ax.plot([], [], lw=1)
        beat_scatter = ax.scatter([], [], s=20)
        txt = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left")
        ax.set_xlabel("Time (s) in buffer")
        ax.set_ylabel("ECG (mV)")
        ax.set_title("LSL ECG stream (rolling)")

    last_refresh = 0.0
    refresh_period = 1.0 / max(1e-6, float(args.refresh_hz))
    t_rate = time.time()
    samples_in_last_sec = 0
    eff_rate = 0.0

    def update_plot():
        nonlocal beats, rhythm_tokens, eff_rate

        y = np.asarray(ecg, dtype=np.float32)
        if y.size == 0:
            return
        x = np.arange(y.size, dtype=np.float32) / float(fs)
        assert ax is not None and line is not None and beat_scatter is not None and txt is not None and fig is not None
        line.set_data(x, y)
        ax.set_xlim(float(x[0]), float(x[-1]))

        # Markers within buffer
        buf_start_global = global_cursor - y.size

        beats = [b for b in beats if b.global_sample >= buf_start_global]
        rx = []
        ry = []
        for b in beats:
            rel = b.global_sample - buf_start_global
            if 0 <= rel < y.size:
                rx.append(float(rel) / float(fs))
                ry.append(float(y[int(rel)]))
        beat_scatter.set_offsets(np.column_stack([rx, ry]) if rx else np.empty((0, 2)))

        rhythm_tokens = [r for r in rhythm_tokens if r.global_sample >= buf_start_global - int(5 * fs)]

        active = rhythm_stack[-1] if rhythm_stack else ""
        active_norm = normalize_rhythm_label(active) if active else "None"
        txt.set_text(
            f"fs={fs:g}Hz  eff_rate~{eff_rate:0.1f}Hz  global_sample={global_cursor}\n"
            f"segment={last_segment or '-'}  rhythm={active_norm}  beats_in_buf={len(beats)}"
        )

        # Color by active rhythm (current)
        bg = {"NSR": "#F3FFF3", "AFIB": "#FFF3F3", "AFL": "#FFFBE6"}.get(active_norm, "#FFFFFF")
        ax.set_facecolor(bg)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    t_start = time.time()
    last_print = 0.0
    while True:
        # Pull ECG data
        chunk, _ts = inlet_ecg.pull_chunk(timeout=0.0, max_samples=4096)
        if chunk:
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            for v in arr.tolist():
                ecg.append(float(v))
            global_cursor += int(arr.size)
            samples_in_last_sec += int(arr.size)

        # Pull markers
        while True:
            m_sample, _m_ts = inlet_ann.pull_sample(timeout=0.0)
            if not m_sample:
                break
            raw = _parse_marker(m_sample)
            if raw is None:
                continue
            kind = str(raw.get("kind", ""))
            if kind == "segment_start":
                last_segment = str(raw.get("record", "")) or last_segment

            gs = raw.get("global_sample")
            try:
                gs_i = int(gs)
            except Exception:
                continue

            symbol = str(raw.get("symbol", ""))
            aux = safe_aux_note(raw.get("aux_note"))

            if symbol and symbol != "+":
                beats.append(BeatMarker(global_sample=gs_i, symbol=symbol))
            if symbol == "+" and aux:
                rhythm_tokens.append(RhythmToken(global_sample=gs_i, aux_note=aux))
                # Update rhythm state machine
                if aux.startswith("("):
                    label = aux[1:]
                    if not label.endswith(")"):
                        if rhythm_stack:
                            rhythm_stack.pop()
                        rhythm_stack.append(label)
                elif aux == ")":
                    if rhythm_stack:
                        rhythm_stack.pop()

        # Effective rate estimate
        now = time.time()
        if now - t_rate >= 1.0:
            eff_rate = float(samples_in_last_sec) / float(now - t_rate)
            t_rate = now
            samples_in_last_sec = 0

        if args.no_plot:
            if now - last_print >= 1.0:
                active = rhythm_stack[-1] if rhythm_stack else ""
                active_norm = normalize_rhythm_label(active) if active else "None"
                print(
                    f"fs={fs:g}Hz eff_rate~{eff_rate:0.1f}Hz global_sample={global_cursor} "
                    f"segment={last_segment or '-'} rhythm={active_norm} beats_cached={len(beats)}"
                )
                last_print = now
        else:
            if now - last_refresh >= refresh_period:
                update_plot()
                last_refresh = now

        if float(args.duration_sec) > 0 and (now - t_start) >= float(args.duration_sec):
            break

        time.sleep(0.005)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
