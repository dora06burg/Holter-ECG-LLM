from __future__ import annotations

"""
Icentia11k (WFDB) helpers used by the streaming simulation.

This module focuses on:
- Resolving the dataset root (handles both .../dataset and .../dataset/Icentia11k)
- Listing records for a patient and sorting by segment id
- Reading ECG waveform (physical units, mV)
- Reading WFDB annotations (.atr)
- True resampling (signal + annotation sample indices)
"""

import os
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal

import wfdb


RE_RECORD_BASENAME = re.compile(r"^p(?P<pid>\d{5})_s(?P<sid>\d+)$")


@dataclass(frozen=True)
class RecordId:
    patient_id: int
    segment_id: int
    record: str  # e.g. "p00000_s00" (relative path under dataset root, no extension)


@dataclass(frozen=True)
class WFDBSegment:
    rec_id: RecordId
    fs: float
    sig_len: int


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

    # Fallback: try one-level subdirs that contain RECORDS
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


def read_records_list(dataset_root: str) -> List[str]:
    records_path = os.path.join(dataset_root, "RECORDS")
    if os.path.isfile(records_path):
        with open(records_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    # Fallback: scan .hea files
    records: List[str] = []
    for root, _, files in os.walk(dataset_root):
        for name in files:
            if not name.endswith(".hea"):
                continue
            rel = os.path.relpath(os.path.join(root, os.path.splitext(name)[0]), dataset_root)
            records.append(rel)
    return sorted(set(records))


def parse_record_id(record: str) -> Optional[RecordId]:
    base = os.path.basename(record)
    m = RE_RECORD_BASENAME.match(base)
    if not m:
        return None
    return RecordId(patient_id=int(m.group("pid")), segment_id=int(m.group("sid")), record=record)


def list_patient_records(dataset_root: str, patient_id: int) -> List[RecordId]:
    ids: List[RecordId] = []
    for rec in read_records_list(dataset_root):
        rid = parse_record_id(rec)
        if rid is None:
            continue
        if rid.patient_id != int(patient_id):
            continue
        ids.append(rid)
    ids.sort(key=lambda x: x.segment_id)
    return ids


def read_segment_header(dataset_root: str, rec: RecordId) -> WFDBSegment:
    header = wfdb.rdheader(os.path.join(dataset_root, rec.record))
    if header.fs is None or header.sig_len is None:
        raise RuntimeError(f"Missing fs/sig_len in header: {rec.record}")
    return WFDBSegment(rec_id=rec, fs=float(header.fs), sig_len=int(header.sig_len))


def read_ecg_mV(dataset_root: str, rec: RecordId, *, channel: int = 0) -> Tuple[np.ndarray, float]:
    """
    Read one WFDB record (single segment) and return ECG in physical units (mV).
    """
    record = wfdb.rdrecord(os.path.join(dataset_root, rec.record), channels=[channel])
    if record.p_signal is None:
        raise RuntimeError(f"Missing p_signal for record: {rec.record}")
    x = record.p_signal[:, 0].astype(np.float32, copy=False)
    return x, float(record.fs)


def read_annotations(dataset_root: str, rec: RecordId):
    ann = wfdb.rdann(os.path.join(dataset_root, rec.record), "atr")
    # ann.sample is ndarray[int]; ann.symbol is list[str]; ann.aux_note is list[str|None]
    return ann


def _fractional_approximation(fs_in: float, fs_out: float, *, max_denominator: int = 10_000) -> Tuple[int, int]:
    """
    Return integers (up, down) such that fs_out ~= fs_in * up / down.
    """
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("fs_in and fs_out must be positive")
    ratio = Fraction.from_float(float(fs_out) / float(fs_in)).limit_denominator(max_denominator)
    return ratio.numerator, ratio.denominator


def resample_signal_polyphase(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    True resampling using polyphase filtering (scipy.signal.resample_poly).
    """
    if abs(fs_out - fs_in) < 1e-9:
        return x.astype(np.float32, copy=False)
    up, down = _fractional_approximation(fs_in, fs_out)
    y = scipy.signal.resample_poly(x, up=up, down=down).astype(np.float32, copy=False)
    return y


def resample_samples(samples_in: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Resample sample indices by time mapping: t = s/fs_in -> s' = round(t*fs_out).
    """
    if abs(fs_out - fs_in) < 1e-9:
        return samples_in.astype(np.int64, copy=False)
    t = samples_in.astype(np.float64) / float(fs_in)
    s_out = np.rint(t * float(fs_out)).astype(np.int64)
    return s_out


def safe_aux_note(aux: Optional[str]) -> str:
    # In this dataset, many aux_note entries are literal string "None".
    if aux is None:
        return ""
    if aux == "None":
        return ""
    return aux


def normalize_beat_symbol(symbol: str) -> str:
    """
    Keep WFDB symbol, but expose a consistent subset used by the dataset:
    N (normal), S (PAC), V (PVC), Q (unknown)
    """
    return symbol


def normalize_rhythm_label(raw_label: str) -> str:
    """
    Map Icentia rhythm tokens to your requested names.
      raw: 'N' -> 'NSR'
      raw: 'AFIB' -> 'AFIB'
      raw: 'AFL' -> 'AFL'
    """
    if raw_label == "N":
        return "NSR"
    return raw_label

