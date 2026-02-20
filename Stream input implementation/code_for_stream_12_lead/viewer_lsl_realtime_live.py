#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一个“更稳”的实时 Viewer：

- 原来的 viewer_lsl_realtime.py 在一些环境里不会弹出窗口，常见原因是：
  - 没有图形界面/没有 DISPLAY（例如 WSL 未开启 WSLg、远程 SSH 无 X11 转发）
  - matplotlib 退化到非交互后端（backend=Agg），因此不会有弹窗，但程序也不会报错

本脚本提供两种显示方式：
1) GUI 模式（有 DISPLAY/Wayland 时）：弹出 matplotlib 窗口实时刷新
2) PNG 模式（无 GUI 时也能用）：把“滚动窗口波形 + 标注”持续写到一个 PNG 文件（覆盖更新）

注意：本脚本不会修改原项目任何逻辑，只是一个单独的新入口文件。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

# 允许从原始 code_for_stream 目录复用工具模块
_THIS_DIR = Path(__file__).resolve().parent
_ORIG_DIR = _THIS_DIR.parent / "code_for_stream"
if _ORIG_DIR.exists() and str(_ORIG_DIR) not in sys.path:
    sys.path.insert(0, str(_ORIG_DIR))

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


# 统一导联顺序（全项目唯一标准）
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Realtime viewer for Icentia LSL ECG + markers (GUI or PNG output).")
    p.add_argument("--lsl-name-ecg", type=str, default="icentia_ecg")
    p.add_argument("--lsl-name-ann", type=str, default="icentia_ann")
    p.add_argument("--resolve-timeout", type=float, default=10.0)

    p.add_argument("--display-sec", type=float, default=12.0, help="How many seconds to display in the rolling window.")
    p.add_argument("--refresh-hz", type=float, default=10.0, help="Refresh rate for GUI/PNG.")
    p.add_argument("--fs", type=float, default=0.0, help="Override sampling rate (0 = read from stream info).")
    p.add_argument("--duration-sec", type=float, default=0.0, help="Run for N seconds then exit (0 = run until Ctrl-C).")

    p.add_argument(
        "--mode",
        choices=["auto", "gui", "png"],
        default="auto",
        help="Display mode. auto=GUI if DISPLAY/Wayland exists, else PNG.",
    )
    p.add_argument(
        "--out-png",
        type=str,
        default=os.path.join("ECG_Model", "stream_result", "lsl_live.png"),
        help="Where to write the rolling PNG when mode=png (or when auto falls back to png).",
    )
    p.add_argument("--dpi", type=int, default=120, help="PNG dpi when mode=png.")
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


def _parse_lead_order(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts if parts else None
    return None


def _has_gui_display() -> bool:
    # Linux/WSL: GUI 依赖 DISPLAY 或 WAYLAND_DISPLAY
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _atomic_write_png(fig, out_path: Path, *, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 注意：matplotlib 会根据文件后缀推断输出格式；若临时文件后缀不是 .png 会报错。
    tmp = out_path.with_name(out_path.name + ".tmp")
    fig.savefig(tmp, dpi=int(dpi), bbox_inches="tight", format="png")
    tmp.replace(out_path)


def _write_autorefresh_html(*, out_png: Path, refresh_ms: int = 200) -> Path:
    """
    生成一个简单的 HTML 页面，用浏览器每隔 refresh_ms 毫秒刷新一次图片（通过 querystring 防缓存）。
    适合在 WSL/无 GUI 环境下：你可以在 Windows 浏览器里打开 http://localhost:PORT/xxx.html 查看实时更新。
    """
    out_html = out_png.with_suffix(".html")
    # 相对路径引用：假设 HTML 与 PNG 在同一目录
    png_name = out_png.name
    html = f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>LSL Live ECG</title>
    <style>
      body {{ margin: 0; font-family: sans-serif; background: #111; color: #eee; }}
      .bar {{ padding: 10px 12px; font-size: 14px; opacity: 0.9; }}
      img {{ display: block; width: 100%; height: auto; }}
      a {{ color: #9cf; }}
    </style>
  </head>
  <body>
    <div class="bar">
      LSL Live ECG (auto-refresh {refresh_ms}ms) · Image: <code>{png_name}</code>
    </div>
    <img id="img" alt="live ecg" />
    <script>
      const img = document.getElementById('img');
      function tick() {{
        img.src = '{png_name}?ts=' + Date.now();
      }}
      tick();
      setInterval(tick, {int(refresh_ms)});
    </script>
  </body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


def main() -> int:
    args = _build_argparser().parse_args()
    pylsl = import_pylsl()

    # --------- 决定显示模式（GUI 或 PNG）---------
    mode = str(args.mode)
    if mode == "auto":
        mode = "gui" if _has_gui_display() else "png"

    # matplotlib 后端必须在 import pyplot 之前设置
    import matplotlib

    if mode == "gui":
        try:
            # TkAgg 通用性较高；但仍要求有 DISPLAY/Wayland
            matplotlib.use("TkAgg", force=True)
        except Exception:
            # 若 GUI 后端不可用，退化到 PNG 模式
            mode = "png"

    if mode == "png":
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    # --------- 连接 LSL ---------
    stream_ecg = _resolve_one_stream(pylsl, name=str(args.lsl_name_ecg), timeout=float(args.resolve_timeout))
    stream_ann = _resolve_one_stream(pylsl, name=str(args.lsl_name_ann), timeout=float(args.resolve_timeout))
    inlet_ecg = pylsl.StreamInlet(stream_ecg, max_buflen=60)
    inlet_ann = pylsl.StreamInlet(stream_ann, max_buflen=60)

    fs = float(args.fs) if float(args.fs) > 0 else float(inlet_ecg.info().nominal_srate())
    if fs <= 0:
        raise SystemExit("ECG stream nominal_srate is not set. Use --fs to override.")

    n_channels = int(inlet_ecg.info().channel_count())
    if n_channels <= 0:
        n_channels = 1

    display_samples = int(round(float(args.display_sec) * fs))
    display_samples = max(10, display_samples)

    ecg_bufs: List[Deque[float]] = [deque(maxlen=display_samples) for _ in range(n_channels)]
    beats: List[BeatMarker] = []
    rhythm_tokens: List[RhythmToken] = []

    global_cursor = 0
    rhythm_stack: List[str] = []
    last_segment: Optional[str] = None
    current_lead_order: List[str] = list(LEAD_ORDER) if n_channels == len(LEAD_ORDER) else [f"ch{i}" for i in range(n_channels)]

    # --------- 初始化图形对象（GUI/PNG 共用一套）---------
    cols = 4 if n_channels >= 4 else n_channels
    rows = int(math.ceil(float(n_channels) / float(cols))) if cols > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8), sharex=True)
    axes_list = np.atleast_1d(axes).ravel().tolist()
    lines = []
    for i in range(n_channels):
        ax = axes_list[i]
        (line,) = ax.plot([], [], lw=1)
        lines.append(line)
        ax.set_title(current_lead_order[i])
        ax.set_ylabel("mV")
        if i >= (n_channels - cols):
            ax.set_xlabel("Time (s)")
    for j in range(n_channels, len(axes_list)):
        axes_list[j].axis("off")
    beat_scatter = axes_list[0].scatter([], [], s=20)
    txt = fig.text(0.01, 0.98, "", va="top", ha="left")
    fig.suptitle("LSL ECG stream (rolling, 12-lead)", x=0.5, y=0.995, fontsize=12)

    if mode == "gui":
        plt.ion()
        # 某些后端需要显式 show 才会真正创建窗口
        plt.show(block=False)

    last_refresh = 0.0
    refresh_period = 1.0 / max(1e-6, float(args.refresh_hz))
    t_rate = time.time()
    samples_in_last_sec = 0
    eff_rate = 0.0

    out_png = Path(str(args.out_png)).expanduser().resolve()
    if mode == "png":
        # 生成一个同名 .html（同目录）方便浏览器自动刷新查看
        _write_autorefresh_html(out_png=out_png, refresh_ms=max(50, int(round(1000.0 / max(1e-6, float(args.refresh_hz))))))

    def _set_axis_titles(order: List[str]) -> None:
        nonlocal current_lead_order
        if not order or len(order) != n_channels:
            return
        if order == current_lead_order:
            return
        current_lead_order = list(order)
        for i in range(n_channels):
            axes_list[i].set_title(current_lead_order[i])

    def update_view() -> None:
        nonlocal beats, rhythm_tokens, eff_rate
        if not ecg_bufs or len(ecg_bufs[0]) == 0:
            return
        buf_len = len(ecg_bufs[0])
        x = np.arange(buf_len, dtype=np.float32) / float(fs)
        x_min = float(x[0])
        x_max = float(x[-1])

        for i in range(n_channels):
            yi = np.asarray(ecg_bufs[i], dtype=np.float32)
            if yi.size == 0:
                continue
            x_i = x if yi.size == buf_len else (np.arange(yi.size, dtype=np.float32) / float(fs))
            lines[i].set_data(x_i, yi)
            ax_i = axes_list[i]
            ax_i.set_xlim(float(x_i[0]), float(x_i[-1]))
            y_min = float(np.min(yi))
            y_max = float(np.max(yi))
            pad = 0.05 * max(1e-6, (y_max - y_min))
            ax_i.set_ylim(y_min - pad, y_max + pad)

        buf_start_global = global_cursor - buf_len

        beats = [b for b in beats if b.global_sample >= buf_start_global]
        rx: List[float] = []
        ry: List[float] = []
        if n_channels > 0:
            y0 = np.asarray(ecg_bufs[0], dtype=np.float32)
            for b in beats:
                rel = b.global_sample - buf_start_global
                if 0 <= rel < y0.size:
                    rx.append(float(rel) / float(fs))
                    ry.append(float(y0[int(rel)]))
        beat_scatter.set_offsets(np.column_stack([rx, ry]) if rx else np.empty((0, 2)))

        rhythm_tokens = [r for r in rhythm_tokens if r.global_sample >= buf_start_global - int(5 * fs)]

        active = rhythm_stack[-1] if rhythm_stack else ""
        active_norm = normalize_rhythm_label(active) if active else "None"
        txt.set_text(
            f"mode={mode} fs={fs:g}Hz  eff_rate~{eff_rate:0.1f}Hz  global_sample={global_cursor}\n"
            f"segment={last_segment or '-'}  rhythm={active_norm}  beats_in_buf={len(beats)}"
        )

        bg = {"NSR": "#F3FFF3", "AFIB": "#FFF3F3", "AFL": "#FFFBE6"}.get(active_norm, "#FFFFFF")
        for i in range(n_channels):
            axes_list[i].set_facecolor(bg)

        if mode == "gui":
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
        else:
            _atomic_write_png(fig, out_png, dpi=int(args.dpi))

    t_start = time.time()
    while True:
        # Pull ECG data
        chunk, _ts = inlet_ecg.pull_chunk(timeout=0.0, max_samples=4096)
        if chunk:
            arr = np.asarray(chunk, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, n_channels) if n_channels > 1 else arr.reshape(-1, 1)
            if arr.shape[1] < n_channels:
                pad = np.zeros((arr.shape[0], n_channels - arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)
            elif arr.shape[1] > n_channels:
                arr = arr[:, :n_channels]
            for i in range(n_channels):
                ecg_bufs[i].extend(arr[:, i].tolist())
            global_cursor += int(arr.shape[0])
            samples_in_last_sec += int(arr.shape[0])

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
                lead_order = _parse_lead_order(raw.get("lead_order"))
                if lead_order:
                    _set_axis_titles(lead_order)
            if kind == "session_start":
                lead_order = _parse_lead_order(raw.get("lead_order"))
                if lead_order:
                    _set_axis_titles(lead_order)

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

        if now - last_refresh >= refresh_period:
            update_view()
            last_refresh = now

        if float(args.duration_sec) > 0 and (now - t_start) >= float(args.duration_sec):
            break

        time.sleep(0.005)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
