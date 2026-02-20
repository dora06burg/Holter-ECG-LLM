# Icentia11k LSL Streaming (Minimal)

This is a minimal, runnable subset for **streaming Icentia11k ECG over LSL** and saving windows as WFDB. It includes:
- **1‑lead baseline** sender/receiver/viewer (`code_for_stream`)
- **12‑lead standardized** sender/receiver/viewer (`code_for_stream_12_lead`)

The 12‑lead version maps Icentia11k's `modified lead I` into standard Lead I and fills all other leads with zeros. It writes `lead_order` and `lead_mask` into `.hea` comments.

## Folder Structure
```
ECG_Model_for_github/
  requirements.txt
  code_for_stream/
    sender_icentia_lsl.py
    receiver_lsl_to_wfdb.py
    viewer_lsl_realtime_live.py
    icentia_wfdb.py
    lsl_utils.py
    vendor/lsl_lib/...
  code_for_stream_12_lead/
    sender_icentia_lsl.py
    receiver_lsl_to_wfdb.py
    viewer_lsl_realtime_live.py
```

## Dependencies
```
python -m pip install -r requirements.txt
```
`lsl_utils.py` can load vendored `liblsl` from `code_for_stream/vendor/lsl_lib/` if your environment lacks system LSL libraries.

## What Each Script Does

**code_for_stream/** (baseline 1‑lead)
- `sender_icentia_lsl.py`: Reads Icentia11k WFDB segments and streams ECG + annotations via LSL.
- `receiver_lsl_to_wfdb.py`: Receives LSL ECG + annotations and writes WFDB windows (`.dat/.hea/.atr`).
- `viewer_lsl_realtime_live.py`: Live viewer (GUI or PNG) for the streaming ECG.
- `icentia_wfdb.py`: WFDB helpers (read, resample, annotations).
- `lsl_utils.py`: LSL import helper with vendored binaries fallback.

**code_for_stream_12_lead/** (standardized 12‑lead)
- `sender_icentia_lsl.py`: Streams 12‑lead data (Lead I valid, others zero) + annotations.
- `receiver_lsl_to_wfdb.py`: Writes 12‑lead WFDB windows and adds `lead_order`/`lead_mask` into `.hea` comments.
- `viewer_lsl_realtime_live.py`: 12‑lead live viewer (12 subplots) with PNG output.

## Run (12‑Lead Recommended)
Use **three terminals** in this exact order.

### 1) Receiver
```
cd /mnt/e/2025grade3/code/ECG_Model_for_github
python code_for_stream_12_lead/receiver_lsl_to_wfdb.py \
  --out-dir /mnt/e/2025grade3/code/ECG_Model_for_github/stream_result/lsl_wfdb_run \
  --window-sec 600 \
  --hop-sec 600 \
  --resolve-timeout 120 \
  --inject-rhythm-state \
  --verbose
```
Note: `--hop-sec` must equal `--window-sec` in this implementation.

### 2) Viewer (PNG mode)
```
cd /mnt/e/2025grade3/code/ECG_Model_for_github
python code_for_stream_12_lead/viewer_lsl_realtime_live.py \
  --mode png \
  --fs 250 \
  --display-sec 12 \
  --refresh-hz 5 \
  --resolve-timeout 120 \
  --out-png /mnt/e/2025grade3/code/ECG_Model_for_github/stream_result/lsl_live.png
```

### 3) Sender
```
cd /mnt/e/2025grade3/code/ECG_Model_for_github
python code_for_stream_12_lead/sender_icentia_lsl.py \
  --dataset /mnt/e/2025grade3/code/ECG_Model/dataset/Icentia11k \
  --patient 0 \
  --segment-start 0 \
  --segment-end 49 \
  --fs-out 250 \
  --chunk-ms 2000 \
  --speed 500 \
  --verbose
```

### Output
- WFDB windows: `<out-dir>/stream_XXXXXX.{dat,hea,atr}`
- Live PNG: `/mnt/e/2025grade3/code/ECG_Model_for_github/stream_result/lsl_live.png`

## Notes
- `.hea` comments include:
  - `lead_order=I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6`
  - `lead_mask=1,0,0,0,0,0,0,0,0,0,0,0`
  - `source_leads=modified_I`, `source_device=CardioSTAT`
- If you interrupt the viewer with Ctrl‑C, matplotlib may print a traceback while saving PNG. Output file is still valid.
