from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from icentia_wfdb import (
    list_patient_records,
    normalize_beat_symbol,
    normalize_rhythm_label,
    parse_record_id,
    read_annotations,
    read_ecg_mV,
    read_segment_header,
    resample_samples,
    resample_signal_polyphase,
    resolve_dataset_root,
    safe_aux_note,
)
from lsl_utils import import_pylsl


@dataclass(frozen=True)
class StreamConfig:
    name_ecg: str
    name_ann: str
    type_ecg: str = "ECG"
    type_ann: str = "Markers"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stream Icentia11k WFDB records as real-time LSL (ECG + annotation markers)."
    )
    p.add_argument("--dataset", type=str, default=os.path.join("..", "dataset"), help="Dataset root (either .../dataset or .../dataset/Icentia11k).")
    p.add_argument("--patient", type=int, default=0, help="Patient id (e.g., 0 for p00000).")

    p.add_argument("--segment-start", type=int, default=0, help="First segment id to stream (inclusive).")
    p.add_argument("--segment-end", type=int, default=49, help="Last segment id to stream (inclusive).")

    p.add_argument("--fs-out", type=float, default=250.0, help="Output sampling rate (Hz). True resampling will be applied if different from input.")
    p.add_argument("--chunk-ms", type=float, default=200.0, help="ECG push chunk size (milliseconds). Smaller = lower latency, higher overhead.")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed. 1.0 = real-time. 10.0 = 10x faster. 0 = as fast as possible.")
    p.add_argument(
        "--watermark-every-chunks",
        type=int,
        default=1,
        help="Send an ANN watermark marker (`kind=ecg_chunk_end`) every N ECG chunks (helps receivers finalize windows correctly even at high speed).",
    )

    p.add_argument("--lsl-name-ecg", type=str, default="icentia_ecg", help="LSL stream name for ECG samples.")
    p.add_argument("--lsl-name-ann", type=str, default="icentia_ann", help="LSL stream name for annotation markers (JSON strings).")
    p.add_argument("--source-id", type=str, default="", help="LSL source_id (optional).")
    p.add_argument(
        "--wait-for-consumers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait until both ECG+ANN outlets have at least one consumer before streaming (prevents missing the beginning at high --speed).",
    )
    p.add_argument(
        "--wait-timeout-sec",
        type=float,
        default=30.0,
        help="Max seconds to wait for LSL consumers when --wait-for-consumers is enabled.",
    )

    p.add_argument("--max-seconds", type=float, default=0.0, help="Stop after streaming this many seconds of *output* ECG (0 = no limit).")
    p.add_argument("--verbose", action="store_true")
    return p


def _lsl_outlets(pylsl, cfg: StreamConfig, *, fs_out: float, source_id: str, patient_id: int, dataset_root: str):
    if not source_id:
        source_id = f"icentia11k_p{patient_id:05d}_fs{fs_out:g}"

    info_ecg = pylsl.StreamInfo(
        name=cfg.name_ecg,
        type=cfg.type_ecg,
        channel_count=1,
        nominal_srate=float(fs_out),
        channel_format="float32",
        source_id=source_id + "_ecg",
    )
    ch = info_ecg.desc().append_child("channels").append_child("channel")
    ch.append_child_value("label", "ecg")
    ch.append_child_value("unit", "mV")
    info_ecg.desc().append_child_value("dataset_root", dataset_root)
    info_ecg.desc().append_child_value("patient_id", f"p{patient_id:05d}")
    info_ecg.desc().append_child_value("fs_out", str(fs_out))

    info_ann = pylsl.StreamInfo(
        name=cfg.name_ann,
        type=cfg.type_ann,
        channel_count=1,
        nominal_srate=0.0,
        channel_format="string",
        source_id=source_id + "_ann",
    )
    info_ann.desc().append_child_value("dataset_root", dataset_root)
    info_ann.desc().append_child_value("patient_id", f"p{patient_id:05d}")
    info_ann.desc().append_child_value("fs_out", str(fs_out))
    info_ann.desc().append_child_value("format", "json")

    outlet_ecg = pylsl.StreamOutlet(info_ecg, chunk_size=0, max_buffered=60)
    outlet_ann = pylsl.StreamOutlet(info_ann, chunk_size=0, max_buffered=60)
    return outlet_ecg, outlet_ann


def _push_marker(outlet_ann, marker: Dict):
    outlet_ann.push_sample([json.dumps(marker, ensure_ascii=False)])


def main() -> int:
    args = _build_argparser().parse_args()
    pylsl = import_pylsl()

    dataset_root = resolve_dataset_root(args.dataset)
    patient_id = int(args.patient)

    patient_records = list_patient_records(dataset_root, patient_id)
    if not patient_records:
        raise SystemExit(f"No records found for patient p{patient_id:05d} under: {dataset_root}")

    # Filter by segment range
    patient_records = [r for r in patient_records if args.segment_start <= r.segment_id <= args.segment_end]
    if not patient_records:
        raise SystemExit("No records after applying --segment-start/--segment-end.")

    # Input fs check (best-effort; segments should share the same fs)
    first_seg = read_segment_header(dataset_root, patient_records[0])
    fs_in_ref = float(first_seg.fs)

    fs_out = float(args.fs_out)
    if fs_out <= 0:
        raise SystemExit("--fs-out must be > 0.")

    cfg = StreamConfig(name_ecg=args.lsl_name_ecg, name_ann=args.lsl_name_ann)
    outlet_ecg, outlet_ann = _lsl_outlets(
        pylsl,
        cfg,
        fs_out=fs_out,
        source_id=str(args.source_id),
        patient_id=patient_id,
        dataset_root=dataset_root,
    )

    chunk_samples = int(round(fs_out * (float(args.chunk_ms) / 1000.0)))
    chunk_samples = max(1, chunk_samples)
    watermark_every = max(1, int(args.watermark_every_chunks))

    if args.wait_for_consumers:
        if args.verbose:
            print("[wait] waiting for LSL consumers (ECG + ANN)...")
        t0 = time.time()
        while True:
            if outlet_ecg.have_consumers() and outlet_ann.have_consumers():
                if args.verbose:
                    print("[wait] consumer connected, starting stream.")
                break
            if (time.time() - t0) > float(args.wait_timeout_sec):
                raise SystemExit(
                    "Timed out waiting for LSL consumers. "
                    "Start the receiver/viewer first, or run with --no-wait-for-consumers."
                )
            time.sleep(0.05)

    global_sample_offset = 0
    max_samples = int(round(float(args.max_seconds) * fs_out)) if args.max_seconds and args.max_seconds > 0 else 0

    _push_marker(
        outlet_ann,
        {
            "kind": "session_start",
            "global_sample": 0,
            "patient_id": f"p{patient_id:05d}",
            "dataset_root": dataset_root,
            "fs_in": fs_in_ref,
            "fs_out": fs_out,
            "chunk_samples": chunk_samples,
            "speed": float(args.speed),
            "segment_start": int(args.segment_start),
            "segment_end": int(args.segment_end),
        },
    )

    t_start = time.time()
    streamed_samples = 0

    for rec in patient_records:
        seg = read_segment_header(dataset_root, rec)
        fs_in = float(seg.fs)
        if abs(fs_in - fs_in_ref) > 1e-6 and args.verbose:
            print(f"[warn] segment {rec.record} fs differs: {fs_in} vs {fs_in_ref}")

        if args.verbose:
            print(f"[seg] start {rec.record} (fs_in={fs_in}, fs_out={fs_out}) global_offset={global_sample_offset}")

        _push_marker(
            outlet_ann,
            {
                "kind": "segment_start",
                "global_sample": int(global_sample_offset),
                "record": rec.record,
                "segment_id": int(rec.segment_id),
                "fs_in": fs_in,
                "fs_out": fs_out,
                "sig_len_in": int(seg.sig_len),
            },
        )

        # Load ECG and annotations
        x_in_mV, fs_in_wave = read_ecg_mV(dataset_root, rec)
        if abs(fs_in_wave - fs_in) > 1e-6 and args.verbose:
            print(f"[warn] rdrecord fs mismatch for {rec.record}: header={fs_in} wave={fs_in_wave}")

        ann = read_annotations(dataset_root, rec)
        ann_samples_out = resample_samples(ann.sample, fs_in=fs_in, fs_out=fs_out)

        # Resample waveform
        x_out_mV = resample_signal_polyphase(x_in_mV, fs_in=fs_in, fs_out=fs_out)

        # Prepare markers for this segment in output sample domain
        markers: List[Dict] = []
        for s_out, symbol, aux in zip(ann_samples_out.tolist(), ann.symbol, ann.aux_note):
            aux = safe_aux_note(aux)
            marker: Dict = {
                "kind": "annotation",
                "global_sample": int(global_sample_offset + int(s_out)),
                "record": rec.record,
                "segment_id": int(rec.segment_id),
                "sample_out_record": int(s_out),
                "symbol": symbol,
            }
            if aux:
                marker["aux_note"] = aux
                if aux.startswith("(") and not aux.endswith(")"):
                    marker["rhythm_label_raw"] = aux[1:]
                    marker["rhythm_label_norm"] = normalize_rhythm_label(aux[1:])
            if symbol != "+":
                marker["beat_symbol_norm"] = normalize_beat_symbol(symbol)
            markers.append(marker)

        markers.sort(key=lambda m: int(m["global_sample"]))
        marker_idx = 0

        # Stream waveform in chunks
        n = int(x_out_mV.shape[0])
        i = 0
        chunk_count = 0
        while i < n:
            if max_samples and streamed_samples >= max_samples:
                break

            chunk = x_out_mV[i : i + chunk_samples]
            if chunk.size == 0:
                break
            # pylsl expects shape (n_samples, n_channels)
            outlet_ecg.push_chunk(chunk.reshape(-1, 1))

            chunk_start_global = global_sample_offset + i
            chunk_end_global = chunk_start_global + int(chunk.shape[0])
            while marker_idx < len(markers) and int(markers[marker_idx]["global_sample"]) < chunk_end_global:
                _push_marker(outlet_ann, markers[marker_idx])
                marker_idx += 1
            chunk_count += 1
            if chunk_count % watermark_every == 0:
                _push_marker(
                    outlet_ann,
                    {
                        "kind": "ecg_chunk_end",
                        "global_sample": int(chunk_end_global),
                    },
                )

            # Pacing
            if float(args.speed) > 0:
                time.sleep(float(chunk.shape[0]) / (fs_out * float(args.speed)))

            i += int(chunk.shape[0])
            streamed_samples += int(chunk.shape[0])

        # If we stop early due to --max-seconds, only advance by the actually-streamed samples.
        streamed_in_segment = i
        global_sample_offset += streamed_in_segment

        # Ensure receivers get a final watermark even if `--watermark-every-chunks > 1`
        # or if we stopped mid-segment.
        _push_marker(
            outlet_ann,
            {
                "kind": "ecg_chunk_end",
                "global_sample": int(global_sample_offset),
            },
        )

        _push_marker(
            outlet_ann,
            {
                "kind": "segment_end",
                "global_sample": int(global_sample_offset),
                "record": rec.record,
                "segment_id": int(rec.segment_id),
                "sig_len_out": int(n),
                "streamed_out": int(streamed_in_segment),
                "truncated": bool(streamed_in_segment < n),
            },
        )

        if max_samples and streamed_samples >= max_samples:
            if args.verbose:
                print("[stop] reached --max-seconds")
            break

    _push_marker(
        outlet_ann,
        {
            "kind": "session_end",
            "global_sample": int(global_sample_offset),
            "elapsed_sec_wall": float(time.time() - t_start),
            "speed": float(args.speed),
        },
    )
    if args.verbose:
        print(f"[done] streamed_samples={streamed_samples} global_end={global_sample_offset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
