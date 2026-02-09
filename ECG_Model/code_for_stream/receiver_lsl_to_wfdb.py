from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import wfdb

from icentia_wfdb import normalize_rhythm_label, safe_aux_note
from lsl_utils import import_pylsl


@dataclass
class MarkerEvent:
    global_sample: int
    symbol: str
    aux_note: str
    raw: Dict[str, Any]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Receive Icentia LSL streams and write them as WFDB windows (.dat/.hea/.atr)."
    )
    p.add_argument("--out-dir", type=str, required=True, help="Output folder for WFDB windows.")
    p.add_argument("--record-prefix", type=str, default="stream", help="Prefix for output WFDB record names.")

    p.add_argument("--lsl-name-ecg", type=str, default="icentia_ecg", help="LSL stream name for ECG.")
    p.add_argument("--lsl-name-ann", type=str, default="icentia_ann", help="LSL stream name for markers.")
    p.add_argument("--resolve-timeout", type=float, default=10.0)

    p.add_argument("--window-sec", type=float, default=10.0, help="Window size in seconds (default: 10).")
    p.add_argument("--hop-sec", type=float, default=10.0, help="Hop size in seconds (default: 10, i.e. no overlap).")
    p.add_argument(
        "--marker-delay-sec",
        type=float,
        default=0.0,
        help="Extra delay (in seconds, in the stream's sample domain) required beyond the window end watermark before finalizing a window. Usually keep 0 when `ecg_chunk_end` watermarks are enabled.",
    )

    p.add_argument("--inject-rhythm-state", action="store_true", help="Inject a rhythm '(XXX' token at sample 0 when the window begins inside a rhythm.")

    p.add_argument("--adc-gain", type=float, default=1000.0, help="WFDB adc_gain for writing .dat/.hea (physical mV -> digital).")
    p.add_argument("--baseline", type=int, default=0, help="WFDB baseline for writing .dat/.hea.")
    p.add_argument("--units", type=str, default="mV")
    p.add_argument("--sig-name", type=str, default="ecg")

    p.add_argument("--max-windows", type=int, default=0, help="Stop after writing N windows (0 = no limit).")
    p.add_argument("--verbose", action="store_true")
    return p


def _resolve_one_stream(pylsl, *, name: str, timeout: float):
    results = pylsl.resolve_byprop("name", name, timeout=timeout)
    if not results:
        raise SystemExit(f"Could not resolve LSL stream with name='{name}' (timeout={timeout}s)")
    return results[0]


def _parse_marker(sample: List[str]) -> Optional[MarkerEvent]:
    if not sample:
        return None
    try:
        raw = json.loads(sample[0])
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    if "global_sample" not in raw:
        return None
    try:
        global_sample = int(raw["global_sample"])
    except Exception:
        return None

    symbol = str(raw.get("symbol", ""))
    aux_note = safe_aux_note(raw.get("aux_note"))
    return MarkerEvent(global_sample=global_sample, symbol=symbol, aux_note=aux_note, raw=raw)


def _update_rhythm_stack(stack: List[str], aux_note: str) -> None:
    # Icentia rhythm encoding:
    # - '(N' / '(AFIB' / '(AFL' starts a region (label without the '(')
    # - ')' ends the most recent region
    if not aux_note:
        return
    if aux_note.startswith("("):
        label = aux_note[1:]
        if label.endswith(")"):
            # Instant token like "(N)" - ignore for state.
            return
        if stack:
            # Defensive: close previous region implicitly by replacing.
            stack.pop()
        stack.append(label)
        return
    if aux_note == ")":
        if stack:
            stack.pop()


def _has_rhythm_start_at_zero(markers_in_window: List[MarkerEvent], window_start_global: int) -> bool:
    for ev in markers_in_window:
        if ev.global_sample != window_start_global:
            continue
        if ev.symbol == "+" and ev.aux_note.startswith("(") and not ev.aux_note.endswith(")"):
            return True
    return False


def _write_wfdb_window(
    *,
    out_dir: Path,
    record_name: str,
    fs: float,
    signal_mV: np.ndarray,
    markers_in_window: List[MarkerEvent],
    window_start_global: int,
    adc_gain: float,
    baseline: int,
    units: str,
    sig_name: str,
    inject_rhythm_state: bool,
    rhythm_stack_state_at_start: List[str],
    markers_meta: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Waveform
    comments = [
        f"global_start_sample={window_start_global}",
        f"fs={fs}",
        "source=LSL",
    ]
    if "source_record_at_start" in markers_meta:
        comments.append(f"source_record_at_start={markers_meta['source_record_at_start']}")
    if "source_segment_id_at_start" in markers_meta:
        comments.append(f"source_segment_id_at_start={markers_meta['source_segment_id_at_start']}")
    if "dataset_root" in markers_meta:
        comments.append(f"dataset_root={markers_meta['dataset_root']}")
    if "patient_id" in markers_meta:
        comments.append(f"patient_id={markers_meta['patient_id']}")
    wfdb.wrsamp(
        record_name,
        fs=fs,
        units=[units],
        sig_name=[sig_name],
        p_signal=signal_mV.reshape(-1, 1).astype(np.float32, copy=False),
        fmt=["16"],
        adc_gain=[float(adc_gain)],
        baseline=[int(baseline)],
        comments=comments,
        write_dir=str(out_dir),
    )

    # Annotations (beat + rhythm tokens). We write window-relative sample indices.
    ann_samples: List[int] = []
    ann_symbols: List[str] = []
    ann_aux: List[str] = []

    # Optional: inject rhythm state at sample 0, only if we are inside a rhythm at window start
    # AND there is no explicit rhythm-start token at sample 0 already.
    if inject_rhythm_state and rhythm_stack_state_at_start:
        if not _has_rhythm_start_at_zero(markers_in_window, window_start_global):
            label = rhythm_stack_state_at_start[-1]
            ann_samples.append(0)
            ann_symbols.append("+")
            ann_aux.append("(" + label)

    for ev in markers_in_window:
        rel = int(ev.global_sample - window_start_global)
        if rel < 0 or rel >= signal_mV.shape[0]:
            continue

        # Only write WFDB-relevant markers.
        if ev.symbol == "":
            continue

        if ev.symbol == "+":
            if not ev.aux_note:
                continue
            ann_samples.append(rel)
            ann_symbols.append("+")
            ann_aux.append(ev.aux_note)
        else:
            ann_samples.append(rel)
            ann_symbols.append(ev.symbol)
            ann_aux.append("")  # keep empty; original dataset often uses literal "None"

    if ann_samples:
        order = np.argsort(np.asarray(ann_samples, dtype=np.int64), kind="stable")
        samples_np = np.asarray([ann_samples[i] for i in order], dtype=np.int64)
        symbols_sorted = [ann_symbols[i] for i in order]
        aux_sorted = [ann_aux[i] for i in order]
        wfdb.wrann(
            record_name,
            "atr",
            sample=samples_np,
            symbol=symbols_sorted,
            aux_note=aux_sorted,
            fs=fs,
            write_dir=str(out_dir),
        )


def main() -> int:
    args = _build_argparser().parse_args()
    pylsl = import_pylsl()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_ecg = _resolve_one_stream(pylsl, name=str(args.lsl_name_ecg), timeout=float(args.resolve_timeout))
    stream_ann = _resolve_one_stream(pylsl, name=str(args.lsl_name_ann), timeout=float(args.resolve_timeout))

    inlet_ecg = pylsl.StreamInlet(stream_ecg, max_buflen=360)
    inlet_ann = pylsl.StreamInlet(stream_ann, max_buflen=360)

    fs = float(inlet_ecg.info().nominal_srate())
    if fs <= 0:
        raise SystemExit("ECG stream nominal_srate is not set. Please set it in the sender.")

    window_samples = int(round(float(args.window_sec) * fs))
    hop_samples = int(round(float(args.hop_sec) * fs))
    if window_samples <= 0 or hop_samples <= 0:
        raise SystemExit("window/hop must be > 0.")
    if hop_samples != window_samples:
        raise SystemExit("This implementation currently expects --hop-sec == --window-sec (no overlap), per your current choice.")

    delay_samples = int(round(float(args.marker_delay_sec) * fs))
    delay_samples = max(0, delay_samples)

    # Buffers
    ecg_buf: List[np.ndarray] = []
    ecg_buf_len = 0
    buf_start_global = 0
    global_cursor = 0

    markers: List[MarkerEvent] = []
    marker_read_count = 0
    marker_kept_count = 0

    rhythm_stack: List[str] = []
    last_watermark_global: Optional[int] = None
    session_meta: Dict[str, Any] = {}
    current_source_record: Optional[str] = None
    current_source_segment_id: Optional[int] = None

    records_path = out_dir / "RECORDS"
    if records_path.exists():
        records_path.unlink()

    written = 0
    last_data_time = time.time()
    session_end_global: Optional[int] = None

    if args.verbose:
        print(f"[recv] fs={fs:g} window_samples={window_samples} watermark_delay_samples={delay_samples}")
        print(f"[recv] writing to: {out_dir}")

    try:
        while True:
            # Pull ECG chunk (non-blocking)
            chunk, _ts = inlet_ecg.pull_chunk(timeout=0.0, max_samples=4096)
            if chunk:
                arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if arr.size:
                    ecg_buf.append(arr)
                    ecg_buf_len += int(arr.size)
                    global_cursor += int(arr.size)
                    last_data_time = time.time()

            # Pull all available markers
            m_chunk, _m_ts = inlet_ann.pull_chunk(timeout=0.0, max_samples=4096)
            if m_chunk:
                for m_sample in m_chunk:
                    ev = _parse_marker(m_sample)
                    if ev is None:
                        continue
                    marker_read_count += 1
                    kind = str(ev.raw.get("kind", ""))
                    if kind == "ecg_chunk_end":
                        last_watermark_global = (
                            ev.global_sample
                            if last_watermark_global is None
                            else max(last_watermark_global, ev.global_sample)
                        )
                        continue
                    if kind == "session_start":
                        # Keep basic context for later embedding into WFDB comments.
                        session_meta = {
                            "dataset_root": ev.raw.get("dataset_root", ""),
                            "patient_id": ev.raw.get("patient_id", ""),
                        }
                        continue
                    if kind == "segment_start":
                        current_source_record = str(ev.raw.get("record", "")) or current_source_record
                        try:
                            current_source_segment_id = int(ev.raw.get("segment_id"))
                        except Exception:
                            current_source_segment_id = current_source_segment_id
                        continue
                    if kind == "session_end":
                        session_end_global = ev.global_sample
                        continue

                    # Keep only actual annotations (beat + rhythm tokens). Segment/session markers are ignored.
                    if kind != "annotation":
                        continue
                    if not ev.symbol:
                        continue
                    markers.append(ev)
                    marker_kept_count += 1

            # If we have enough ECG to finalize a window AND the marker stream watermark
            # tells us that all markers up to the window end have been delivered.
            window_end_global = buf_start_global + window_samples
            watermark_target = window_end_global + delay_samples
            while (
                ecg_buf_len >= window_samples
                and (last_watermark_global is not None)
                and (last_watermark_global >= watermark_target)
            ):
                # Materialize first window_samples from ecg_buf
                window_parts: List[np.ndarray] = []
                remaining = window_samples
                while remaining > 0 and ecg_buf:
                    head = ecg_buf[0]
                    if head.size <= remaining:
                        window_parts.append(head)
                        remaining -= int(head.size)
                        ecg_buf.pop(0)
                    else:
                        window_parts.append(head[:remaining])
                        ecg_buf[0] = head[remaining:]
                        remaining = 0
                window = np.concatenate(window_parts, axis=0)
                assert window.shape[0] == window_samples
                ecg_buf_len -= window_samples

                window_start_global = buf_start_global
                window_end_global = buf_start_global + window_samples

                # Extract markers for this window (keep list sorted-ish)
                # Note: markers list may not be strictly sorted due to transport; sort within a small bound.
                markers.sort(key=lambda e: e.global_sample)
                in_window: List[MarkerEvent] = []
                while markers and markers[0].global_sample < window_start_global:
                    # Update rhythm state for anything strictly before the window.
                    ev0 = markers.pop(0)
                    if ev0.symbol == "+" and ev0.aux_note:
                        _update_rhythm_stack(rhythm_stack, ev0.aux_note)

                # Snapshot state at window start (for injection)
                rhythm_state_at_start = list(rhythm_stack)

                while markers and markers[0].global_sample < window_end_global:
                    ev0 = markers.pop(0)
                    in_window.append(ev0)
                    if ev0.symbol == "+" and ev0.aux_note:
                        _update_rhythm_stack(rhythm_stack, ev0.aux_note)

                record_name = f"{args.record_prefix}_{written:06d}"
                markers_meta: Dict[str, Any] = dict(session_meta)
                if current_source_record:
                    markers_meta["source_record_at_start"] = current_source_record
                if current_source_segment_id is not None:
                    markers_meta["source_segment_id_at_start"] = current_source_segment_id
                _write_wfdb_window(
                    out_dir=out_dir,
                    record_name=record_name,
                    fs=fs,
                    signal_mV=window,
                    markers_in_window=in_window,
                    window_start_global=window_start_global,
                    adc_gain=float(args.adc_gain),
                    baseline=int(args.baseline),
                    units=str(args.units),
                    sig_name=str(args.sig_name),
                    inject_rhythm_state=bool(args.inject_rhythm_state),
                    rhythm_stack_state_at_start=rhythm_state_at_start,
                    markers_meta=markers_meta,
                )
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(record_name + "\n")

                written += 1
                buf_start_global += hop_samples
                if args.verbose:
                    active = rhythm_state_at_start[-1] if rhythm_state_at_start else "None"
                    print(
                        f"[write] {record_name} start={window_start_global} active_rhythm={normalize_rhythm_label(active) if active!='None' else 'None'} markers={len(in_window)}"
                    )

                if args.max_windows and written >= int(args.max_windows):
                    if args.verbose:
                        print("[stop] reached --max-windows")
                    return 0

            # Stop condition: if sender ended and we have written all full windows and no new data for a bit.
            if session_end_global is not None:
                # If no new data recently and not enough buffered for another full window, stop.
                if (time.time() - last_data_time) > 2.0 and ecg_buf_len < window_samples:
                    if args.verbose:
                        print(
                            f"[stop] session_end received and buffers drained. markers_read={marker_read_count} markers_kept={marker_kept_count} watermark={last_watermark_global}"
                        )
                    break

            time.sleep(0.01)

    except KeyboardInterrupt:
        if args.verbose:
            print("[stop] KeyboardInterrupt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
