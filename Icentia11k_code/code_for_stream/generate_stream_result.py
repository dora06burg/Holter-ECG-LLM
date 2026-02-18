from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DemoConfig:
    name: str
    fs_out: float
    seconds: float
    windows: int
    speed: float


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a full LSL streaming demo and generate a report under ECG_Model/stream_result.")
    p.add_argument("--dataset", type=str, default=os.path.join("..", "dataset"))
    p.add_argument("--patient", type=int, default=0)
    p.add_argument("--chunk-ms", type=float, default=200.0)
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--hop-sec", type=float, default=10.0)
    p.add_argument("--inject-rhythm-state", action="store_true", default=True)

    p.add_argument("--run-resample-demo", action="store_true", help="Also run a second demo with fs_out=200Hz to showcase true resampling.")
    p.add_argument("--speed", type=float, default=20.0, help="Playback speed for the main demo.")
    p.add_argument("--out-root", type=str, default=os.path.join("..", "stream_result"), help="Output root folder (default: ECG_Model/stream_result).")
    p.add_argument("--windows", type=int, default=3, help="Number of 10s windows to record for the main demo.")
    return p


def _run(cmd: List[str], *, cwd: Path, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        return subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)


def _wait(proc: subprocess.Popen, *, timeout_sec: float) -> int:
    try:
        return int(proc.wait(timeout=timeout_sec))
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            return int(proc.wait(timeout=5))
        except subprocess.TimeoutExpired:
            proc.kill()
            return int(proc.wait(timeout=5))


def _write_report(report_dir: Path, *, run_meta: Dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    md = report_dir / "README.md"

    lines: List[str] = []
    lines.append("# Icentia11k LSL 流式模拟：运行结果\n")
    lines.append("本报告由 `ECG_Model/code_for_stream/generate_stream_result.py` 自动生成。\n")

    lines.append("## 目录结构\n")
    lines.append("- `wfdb_250Hz/`：Receiver 写回的 WFDB 窗口（`.dat/.hea/.atr` + `RECORDS`）\n")
    if run_meta.get("resample_demo"):
        lines.append("- `wfdb_200Hz/`：同上，但 Sender 做了 250→200Hz 真重采样\n")
    lines.append("- `report/assets/`：窗口波形 + 标注可视化 PNG\n")
    lines.append("- `logs/`：sender/receiver 的运行日志\n")

    lines.append("\n## 本次运行参数（关键）\n")
    lines.append("```json")
    lines.append(json.dumps(run_meta, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## 可视化结果（示例）\n")
    lines.append("### 250Hz 窗口\n")
    lines.append("下面每张图对应一个 10 秒窗口，背景色表示节律（NSR/AFIB/AFL），散点表示 beat 标注。\n")
    lines.append("（注意：NSR 的起始 token 在 Icentia11k 里通常是 `+( '(N')`，报告里会显示为 NSR。）\n")
    lines.append("")
    lines.append("![](assets/250Hz/summary.png)\n")

    if run_meta.get("resample_demo"):
        lines.append("### 200Hz（真重采样演示）\n")
        lines.append("这组结果展示 `--fs-out 200` 时：波形与 `.atr` 的 sample index 都按时间映射到 200Hz。\n")
        lines.append("")
        lines.append("![](assets/200Hz/summary.png)\n")

    lines.append("## 快速上手（你自己复现）\n")
    lines.append("推荐按以下顺序启动，避免高倍速时丢开头：\n")
    lines.append("1. （可选）实时 Viewer：`python ECG_Model/code_for_stream/viewer_lsl_realtime.py`\n")
    lines.append("2. Receiver（写回 WFDB）：`python ECG_Model/code_for_stream/receiver_lsl_to_wfdb.py --out-dir ... --inject-rhythm-state --verbose`\n")
    lines.append("3. Sender（播放并推流）：`python ECG_Model/code_for_stream/sender_icentia_lsl.py --dataset ECG_Model/dataset --patient 0 --speed 20 --verbose`\n")
    lines.append("")
    lines.append("更详细说明见：`ECG_Model/code_for_stream/README.md`。\n")

    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _build_argparser().parse_args()
    here = Path(__file__).resolve().parent
    ecg_model_dir = here.parent
    out_root = (here / args.out_root).resolve() if not os.path.isabs(args.out_root) else Path(args.out_root).resolve()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_id}"
    logs_dir = run_dir / "logs"
    report_dir = run_dir / "report"

    wfdb_250_dir = run_dir / "wfdb_250Hz"
    wfdb_200_dir = run_dir / "wfdb_200Hz"
    assets_250 = report_dir / "assets" / "250Hz"
    assets_200 = report_dir / "assets" / "200Hz"

    run_dir.mkdir(parents=True, exist_ok=True)

    # Demo configs
    window_sec = float(args.window_sec)
    windows_main = int(args.windows)
    seconds_main = window_sec * windows_main
    demos: List[DemoConfig] = [
        DemoConfig(name="250Hz", fs_out=250.0, seconds=seconds_main, windows=windows_main, speed=float(args.speed)),
    ]
    if args.run_resample_demo:
        demos.append(DemoConfig(name="200Hz", fs_out=200.0, seconds=window_sec * 1, windows=1, speed=float(args.speed)))

    for demo in demos:
        wfdb_out = wfdb_250_dir if demo.name == "250Hz" else wfdb_200_dir
        assets_out = assets_250 if demo.name == "250Hz" else assets_200

        # Sender first (creates outlets, then waits for consumers)
        sender_log = logs_dir / f"sender_{demo.name}.log"
        receiver_log = logs_dir / f"receiver_{demo.name}.log"

        sender_cmd = [
            "python",
            str(here / "sender_icentia_lsl.py"),
            "--dataset",
            str((ecg_model_dir / "dataset").resolve() if args.dataset == os.path.join("..", "dataset") else Path(args.dataset).resolve()),
            "--patient",
            str(int(args.patient)),
            "--fs-out",
            str(float(demo.fs_out)),
            "--chunk-ms",
            str(float(args.chunk_ms)),
            "--speed",
            str(float(demo.speed)),
            "--max-seconds",
            str(float(demo.seconds)),
            "--verbose",
        ]

        receiver_cmd = [
            "python",
            str(here / "receiver_lsl_to_wfdb.py"),
            "--out-dir",
            str(wfdb_out),
            "--window-sec",
            str(float(args.window_sec)),
            "--hop-sec",
            str(float(args.hop_sec)),
            "--max-windows",
            str(int(demo.windows)),
            "--verbose",
        ]
        if args.inject_rhythm_state:
            receiver_cmd.append("--inject-rhythm-state")

        sender = _run(sender_cmd, cwd=here, log_path=sender_log)
        time.sleep(0.5)
        receiver = _run(receiver_cmd, cwd=here, log_path=receiver_log)

        # Wait for receiver (it should finish after max-windows)
        _wait(receiver, timeout_sec=60.0)
        _wait(sender, timeout_sec=60.0)

        # Plot WFDB windows to PNG
        plot_log = logs_dir / f"plot_{demo.name}.log"
        plot_cmd = [
            "python",
            str(here / "plot_wfdb_windows.py"),
            "--wfdb-dir",
            str(wfdb_out),
            "--out-dir",
            str(assets_out),
            "--summary-n",
            str(int(demo.windows)),
            "--dpi",
            "150",
        ]
        plot = _run(plot_cmd, cwd=here, log_path=plot_log)
        _wait(plot, timeout_sec=60.0)

    run_meta: Dict = {
        "run_id": run_id,
        "patient": int(args.patient),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "chunk_ms": float(args.chunk_ms),
        "inject_rhythm_state": bool(args.inject_rhythm_state),
        "demos": [demo.__dict__ for demo in demos],
        "resample_demo": bool(args.run_resample_demo),
        "paths": {
            "run_dir": str(run_dir),
            "wfdb_250Hz": str(wfdb_250_dir),
            "wfdb_200Hz": str(wfdb_200_dir) if args.run_resample_demo else "",
            "report_dir": str(report_dir),
            "logs_dir": str(logs_dir),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report_dir, run_meta=run_meta)

    # Convenience output: also mirror the latest report under out_root/report.
    # On some mounted filesystems, copying metadata can raise EPERM/permission errors,
    # so we copy *files* without metadata (copyfile) and fall back to a pointer README.
    (out_root / "LATEST_RUN.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    latest_report_dir = out_root / "report"
    prev_dir = out_root / "report_prev"
    try:
        if latest_report_dir.exists():
            if prev_dir.exists():
                shutil.rmtree(prev_dir)
            try:
                latest_report_dir.rename(prev_dir)
            except OSError:
                shutil.rmtree(latest_report_dir)
        shutil.copytree(
            report_dir,
            latest_report_dir,
            copy_function=shutil.copyfile,
        )
    except Exception:
        latest_report_dir.mkdir(parents=True, exist_ok=True)
        (latest_report_dir / "README.md").write_text(
            "# Latest report\n\n"
            f"Latest run report is located at:\n\n- `{report_dir}`\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
