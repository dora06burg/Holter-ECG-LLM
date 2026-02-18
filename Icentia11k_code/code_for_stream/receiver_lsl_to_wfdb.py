#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 这个脚本做什么？
# - 接收两路 LSL（Lab Streaming Layer）流：
#   1) ECG 波形流（默认名：icentia_ecg）
#   2) 标注/事件流（默认名：icentia_ann），里面包含 beat / rhythm 等 annotation
# - 按固定时间窗口（window-sec）把 ECG 切片写成 WFDB 记录：
#   - 波形：.dat + .hea（wfdb.wrsamp）
#   - 标注：.atr（wfdb.wrann）
# - 并生成一个 RECORDS 文件，列出所有写出的 record_name（方便 WFDB 工具读取目录）
#
# 快速上手（示例）：
# - 先启动发送端（例如 sender_icentia_lsl.py），确保它在推送 icentia_ecg / icentia_ann
# - 再运行本脚本：
#   python receiver_lsl_to_wfdb.py --out-dir ./out --verbose
# - 输出会在 ./out 下，形如 stream_000000.dat/.hea/.atr

# 未来导入：让类型注解在运行时不立即求值（常用于前向引用/降低开销）
from __future__ import annotations

# argparse：解析命令行参数（--out-dir 等）
import argparse
# json：解析 marker 流里发来的 JSON 字符串
import json
# os：操作系统相关工具（此文件里目前未直接用到，但保留）
import os
# time：计时/睡眠/超时判断
import time
# dataclass：用更简洁的方式定义“数据结构类”
from dataclasses import dataclass
# Path：更现代的路径处理（比 os.path 更直观）
from pathlib import Path
# typing：类型标注（提高可读性与 IDE 体验）
from typing import Any, Dict, List, Optional, Tuple

# numpy：高效的数值数组；用于拼接/排序/类型转换
import numpy as np

# wfdb：WFDB（波形数据库）读写库；这里用于写 .dat/.hea/.atr
import wfdb

# icentia_wfdb：本项目里关于 Icentia/WFDB 的辅助函数
from icentia_wfdb import normalize_rhythm_label, safe_aux_note
# lsl_utils：延迟导入 pylsl（避免环境缺依赖时直接崩溃，并给出更友好提示）
from lsl_utils import import_pylsl


# 一个 marker 事件在本脚本中的内部表示（从 LSL 标注流解析出来）
@dataclass
class MarkerEvent:
    # global_sample：全局采样点编号（以 ECG 采样点为“时间轴”对齐的整数）
    global_sample: int
    # symbol：WFDB annotation 的 symbol（例如 '+' 表示 aux_note token；或 'N' 等 beat 符号）
    symbol: str
    # aux_note：WFDB annotation 的 aux_note（常用于节律 token，比如 '(AFIB'、')' 等）
    aux_note: str
    # raw：原始 JSON dict（保留完整信息，便于调试/扩展）
    raw: Dict[str, Any]


# 构建命令行参数解析器（把脚本的可配置项集中在一起）
def _build_argparser() -> argparse.ArgumentParser:
    # 创建 ArgumentParser，并提供一段简介
    p = argparse.ArgumentParser(
        description="Receive Icentia LSL streams and write them as WFDB windows (.dat/.hea/.atr)."
    )
    # 输出目录：写 WFDB 文件到哪里（必填）
    p.add_argument("--out-dir", type=str, required=True, help="Output folder for WFDB windows.")
    # record 前缀：生成的 WFDB record_name 会是 prefix_000000 这种
    p.add_argument("--record-prefix", type=str, default="stream", help="Prefix for output WFDB record names.")

    # ECG LSL 流的名字（需要和发送端保持一致）
    p.add_argument("--lsl-name-ecg", type=str, default="icentia_ecg", help="LSL stream name for ECG.")
    # 标注 LSL 流的名字（需要和发送端保持一致）
    p.add_argument("--lsl-name-ann", type=str, default="icentia_ann", help="LSL stream name for markers.")
    # resolve 超时时间：在 LSL 网络里找流，最多等多久（秒）
    p.add_argument("--resolve-timeout", type=float, default=10.0)

    # window-sec：每个输出 WFDB 记录包含多少秒的 ECG
    p.add_argument("--window-sec", type=float, default=10.0, help="Window size in seconds (default: 10).")
    # hop-sec：窗口滑动步长；这里要求 hop==window（无重叠）
    p.add_argument("--hop-sec", type=float, default=10.0, help="Hop size in seconds (default: 10, i.e. no overlap).")
    # marker-delay-sec：为了等“标注晚到”的情况，可在窗口末尾额外等一段再落盘
    p.add_argument(
        "--marker-delay-sec",
        type=float,
        default=0.0,
        help="Extra delay (in seconds, in the stream's sample domain) required beyond the window end watermark before finalizing a window. Usually keep 0 when `ecg_chunk_end` watermarks are enabled.",
    )

    # inject-rhythm-state：当窗口开始时正处于某个节律区间里，是否在 sample 0 注入一个 '(XXX' token
    p.add_argument("--inject-rhythm-state", action="store_true", help="Inject a rhythm '(XXX' token at sample 0 when the window begins inside a rhythm.")

    # adc-gain/baseline/units/sig-name：写 WFDB 头文件 .hea 时需要的信号元信息
    # - 这里假设 signal_mV 的单位是 mV（物理量），写出时会乘 adc_gain 转成数字量
    p.add_argument("--adc-gain", type=float, default=1000.0, help="WFDB adc_gain for writing .dat/.hea (physical mV -> digital).")
    p.add_argument("--baseline", type=int, default=0, help="WFDB baseline for writing .dat/.hea.")
    p.add_argument("--units", type=str, default="mV")
    p.add_argument("--sig-name", type=str, default="ecg")

    # max-windows：写出多少个窗口后自动停止（0 表示不限制）
    p.add_argument("--max-windows", type=int, default=0, help="Stop after writing N windows (0 = no limit).")
    # verbose：打印更多运行信息，便于调试
    p.add_argument("--verbose", action="store_true")
    # 返回构建好的 parser
    return p


# 根据 LSL 的 stream name 解析（resolve）出一个流（StreamInfo）
def _resolve_one_stream(pylsl, *, name: str, timeout: float):
    # 按属性 name 查找 LSL 流（timeout 秒内找不到就返回空）
    results = pylsl.resolve_byprop("name", name, timeout=timeout)
    # 如果没找到，直接退出（SystemExit 会让脚本以非 0 方式结束并打印信息）
    if not results:
        raise SystemExit(f"Could not resolve LSL stream with name='{name}' (timeout={timeout}s)")
    # 只取第一个匹配的流（一般同名只会有一个）
    return results[0]


# 解析标注流里的一条样本（通常是一个字符串列表，sample[0] 是 JSON 字符串）
def _parse_marker(sample: List[str]) -> Optional[MarkerEvent]:
    # 空样本直接忽略
    if not sample:
        return None
    # 尝试把 sample[0] 当 JSON 解析
    try:
        raw = json.loads(sample[0])
    except Exception:
        # 解析失败：忽略该 marker
        return None
    # 我们期望 raw 是 dict（JSON object）
    if not isinstance(raw, dict):
        return None

    # 必须有 global_sample 字段，用于和 ECG 样本对齐
    if "global_sample" not in raw:
        return None
    # global_sample 必须能转成 int
    try:
        global_sample = int(raw["global_sample"])
    except Exception:
        return None

    # symbol：WFDB 的标注符号（beat 或 '+'）
    symbol = str(raw.get("symbol", ""))
    # aux_note：附加说明（用 safe_aux_note 做清洗/截断，防止 WFDB 写入异常）
    aux_note = safe_aux_note(raw.get("aux_note"))
    # 打包成我们内部的 MarkerEvent
    return MarkerEvent(global_sample=global_sample, symbol=symbol, aux_note=aux_note, raw=raw)


# 维护“当前所处节律区间”的栈（stack[-1] 代表最近一次开始但尚未结束的节律）
def _update_rhythm_stack(stack: List[str], aux_note: str) -> None:
    # 中文解释：Icentia 的节律 token 编码规则大致是
    # - '(N' / '(AFIB' / '(AFL' 之类表示“进入某节律区间”（不带右括号）
    # - ')' 表示“退出最近的节律区间”
    # Icentia rhythm encoding:
    # - '(N' / '(AFIB' / '(AFL' starts a region (label without the '(')
    # - ')' ends the most recent region
    # aux_note 为空则无事可做
    if not aux_note:
        return
    # 以 '(' 开头：表示节律开始 token
    if aux_note.startswith("("):
        # 去掉左括号，得到 label（如 "AFIB"）
        label = aux_note[1:]
        # 如果 label 以 ')' 结尾，形如 "(N)" 这种瞬时 token，这里不作为“区间状态”处理
        if label.endswith(")"):
            # Instant token like "(N)" - ignore for state.
            return
        # 如果栈里已有一个未关闭的节律，出于防御性考虑，直接替换（隐式关闭旧的）
        if stack:
            # Defensive: close previous region implicitly by replacing.
            stack.pop()
        # 把新的节律 label 入栈
        stack.append(label)
        return
    # aux_note 仅为 ')'：表示节律区间结束
    if aux_note == ")":
        # 有开始才有结束；栈非空就弹出
        if stack:
            stack.pop()


# 判断窗口起点是否已经存在“显式的节律开始 token”
# 目的：避免 inject-rhythm-state 时重复写一个 '(XXX' 在 sample 0
def _has_rhythm_start_at_zero(markers_in_window: List[MarkerEvent], window_start_global: int) -> bool:
    # 遍历窗口内所有 marker
    for ev in markers_in_window:
        # 只关心发生在窗口起点（global_sample == window_start_global）的事件
        if ev.global_sample != window_start_global:
            continue
        # 节律开始 token 的典型形式：symbol='+' 且 aux_note 以 '(' 开头且不以 ')' 结尾（表示区间开始）
        if ev.symbol == "+" and ev.aux_note.startswith("(") and not ev.aux_note.endswith(")"):
            return True
    # 没找到则返回 False
    return False


# 把一个窗口写成 WFDB 记录（波形 + 标注）
def _write_wfdb_window(
    # 这里使用关键字参数调用，避免传参顺序出错（可读性更好）
    *,
    # out_dir：输出目录
    out_dir: Path,
    # record_name：WFDB 记录名（不含扩展名）
    record_name: str,
    # fs：采样率（Hz）
    fs: float,
    # signal_mV：窗口内的 ECG 信号（单位 mV，1D 数组，长度=window_samples）
    signal_mV: np.ndarray,
    # markers_in_window：窗口内的 markers（global_sample 坐标）
    markers_in_window: List[MarkerEvent],
    # window_start_global：窗口起点的 global_sample（用于把 marker 转成窗口内相对采样点）
    window_start_global: int,
    # adc_gain/baseline/units/sig_name：WFDB 头文件所需元信息
    adc_gain: float,
    baseline: int,
    units: str,
    sig_name: str,
    # inject_rhythm_state：是否在窗口开头注入节律状态（见 main 的参数说明）
    inject_rhythm_state: bool,
    # rhythm_stack_state_at_start：窗口开始时的节律栈快照（用于注入）
    rhythm_stack_state_at_start: List[str],
    # markers_meta：一些上下文信息（dataset_root/patient_id 等），会写入 WFDB comments
    markers_meta: Dict[str, Any],
) -> None:
    # 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 写波形（.dat + .hea）----
    # Waveform
    # WFDB .hea 里支持 comments 字段；这里写一些便于溯源/调试的键值对
    comments = [
        # 该窗口在“全局采样轴”上的起点
        f"global_start_sample={window_start_global}",
        # 采样率
        f"fs={fs}",
        # 数据来源标记
        "source=LSL",
    ]
    # 如果 meta 里带有 source_record（原始数据文件名），也写入 comment
    if "source_record_at_start" in markers_meta:
        comments.append(f"source_record_at_start={markers_meta['source_record_at_start']}")
    # 如果 meta 里带有 source_segment_id（原始数据分段 id），也写入 comment
    if "source_segment_id_at_start" in markers_meta:
        comments.append(f"source_segment_id_at_start={markers_meta['source_segment_id_at_start']}")
    # dataset_root：原始数据根目录（如果发送端提供）
    if "dataset_root" in markers_meta:
        comments.append(f"dataset_root={markers_meta['dataset_root']}")
    # patient_id：病人/记录 id（如果发送端提供）
    if "patient_id" in markers_meta:
        comments.append(f"patient_id={markers_meta['patient_id']}")
    # wfdb.wrsamp：写一个采样记录（samples）
    wfdb.wrsamp(
        # record 名（不含扩展名）
        record_name,
        # 采样率
        fs=fs,
        # 每个通道的单位（这里只有 1 通道）
        units=[units],
        # 每个通道的名字（这里只有 1 通道）
        sig_name=[sig_name],
        # 物理信号矩阵：形状 (n_samples, n_channels)，这里 reshape 成 (N,1)
        p_signal=signal_mV.reshape(-1, 1).astype(np.float32, copy=False),
        # 存储格式：这里写 16-bit 整型（WFDB 常见格式之一）
        fmt=["16"],
        # adc_gain：物理量到数字量的比例（每通道一个）
        adc_gain=[float(adc_gain)],
        # baseline：基线偏移（每通道一个）
        baseline=[int(baseline)],
        # 写入头文件的 comments
        comments=comments,
        # 输出目录（wfdb 接口需要 str）
        write_dir=str(out_dir),
    )

    # ---- 2) 写标注（.atr）----
    # Annotations (beat + rhythm tokens). We write window-relative sample indices.
    # ann_samples：标注在“窗口内相对采样点”的位置（整数）
    ann_samples: List[int] = []
    # ann_symbols：对应每个标注的 symbol
    ann_symbols: List[str] = []
    # ann_aux：对应每个标注的 aux_note（对 '+' 类型才有意义）
    ann_aux: List[str] = []

    # Optional: inject rhythm state at sample 0, only if we are inside a rhythm at window start
    # AND there is no explicit rhythm-start token at sample 0 already.
    # 可选逻辑：如果窗口开始时已经“处于某节律区间内”，但窗口里又没有在 sample 0 显式写 '(XXX'，
    # 那就人为补一个 token，保证单窗口独立读取时也知道“从一开始就在 AFIB 等节律里”
    if inject_rhythm_state and rhythm_stack_state_at_start:
        if not _has_rhythm_start_at_zero(markers_in_window, window_start_global):
            # 栈顶就是当前激活的节律 label
            label = rhythm_stack_state_at_start[-1]
            # 注入到窗口内 sample=0
            ann_samples.append(0)
            ann_symbols.append("+")
            # 注意：这里写 '(LABEL'（不带右括号），表示节律区间开始
            ann_aux.append("(" + label)

    # 遍历窗口内所有 marker，写入 WFDB annotation 列表
    for ev in markers_in_window:
        # 把 global_sample 转成“相对窗口起点”的采样点
        rel = int(ev.global_sample - window_start_global)
        # 不在窗口范围内的事件丢弃（防御性判断）
        if rel < 0 or rel >= signal_mV.shape[0]:
            continue

        # Only write WFDB-relevant markers.
        # symbol 为空说明不写入
        if ev.symbol == "":
            continue

        # '+' 表示这条 annotation 的含义在 aux_note 里（比如 '(AFIB' 或 ')')
        if ev.symbol == "+":
            # 没有 aux_note 就没法表达 token，跳过
            if not ev.aux_note:
                continue
            # 写入样本点、符号、aux_note
            ann_samples.append(rel)
            ann_symbols.append("+")
            ann_aux.append(ev.aux_note)
        else:
            # 其他 symbol 视为 beat 等事件（aux_note 留空）
            ann_samples.append(rel)
            ann_symbols.append(ev.symbol)
            ann_aux.append("")  # keep empty; original dataset often uses literal "None"

    # 如果这个窗口内确实有标注，才写 .atr（无标注就不生成 .atr）
    if ann_samples:
        # 为了保险起见按 sample 排序（稳定排序，保持相同 sample 的原有顺序）
        order = np.argsort(np.asarray(ann_samples, dtype=np.int64), kind="stable")
        # 排序后的 sample 序列
        samples_np = np.asarray([ann_samples[i] for i in order], dtype=np.int64)
        # 排序后的 symbol 列表
        symbols_sorted = [ann_symbols[i] for i in order]
        # 排序后的 aux_note 列表
        aux_sorted = [ann_aux[i] for i in order]
        # wfdb.wrann：写 annotation 文件（这里 annotation type 用 "atr"）
        wfdb.wrann(
            # record 名（不含扩展名）
            record_name,
            # annotation 扩展名/类型
            "atr",
            # 样本点数组
            sample=samples_np,
            # 符号列表（长度与 sample 一致）
            symbol=symbols_sorted,
            # aux_note 列表（长度与 sample 一致）
            aux_note=aux_sorted,
            # 采样率（写入 annotation 头信息用）
            fs=fs,
            # 输出目录
            write_dir=str(out_dir),
        )


# 程序入口：不断从 LSL 拉取 ECG/标注，按窗口落盘成 WFDB
def main() -> int:
    # 解析命令行参数
    args = _build_argparser().parse_args()
    # 导入 pylsl（如果未安装/环境不对，这里会报更清晰的错误）
    pylsl = import_pylsl()

    # 规范化输出路径，并创建目录
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析（resolve）两路 LSL 流：ECG + annotation
    stream_ecg = _resolve_one_stream(pylsl, name=str(args.lsl_name_ecg), timeout=float(args.resolve_timeout))
    stream_ann = _resolve_one_stream(pylsl, name=str(args.lsl_name_ann), timeout=float(args.resolve_timeout))

    # 为两路流创建 inlet（接收端）；max_buflen 是 LSL 的内部缓冲时长（秒）
    inlet_ecg = pylsl.StreamInlet(stream_ecg, max_buflen=360)
    inlet_ann = pylsl.StreamInlet(stream_ann, max_buflen=360)

    # 从 ECG 流的 StreamInfo 里读取 nominal_srate（采样率），必须由发送端设置
    fs = float(inlet_ecg.info().nominal_srate())
    if fs <= 0:
        raise SystemExit("ECG stream nominal_srate is not set. Please set it in the sender.")

    # 把 window/hop 从秒换算成采样点数
    window_samples = int(round(float(args.window_sec) * fs))
    hop_samples = int(round(float(args.hop_sec) * fs))
    # 基本参数检查
    if window_samples <= 0 or hop_samples <= 0:
        raise SystemExit("window/hop must be > 0.")
    # 当前实现只支持不重叠窗口（hop == window）
    if hop_samples != window_samples:
        raise SystemExit("This implementation currently expects --hop-sec == --window-sec (no overlap), per your current choice.")

    # marker_delay_sec 同样换算成采样点（用于 watermark_target）
    delay_samples = int(round(float(args.marker_delay_sec) * fs))
    # delay 不允许为负
    delay_samples = max(0, delay_samples)

    # Buffers
    # ecg_buf：用列表保存多个 numpy 小块（chunk），避免频繁大数组拼接
    ecg_buf: List[np.ndarray] = []
    # ecg_buf_len：当前缓冲区累计的样本点数
    ecg_buf_len = 0
    # buf_start_global：当前缓冲区最前面的样本对应的 global_sample（窗口起点会从这里走）
    buf_start_global = 0
    # global_cursor：累计接收过的样本数（这里主要用于计数/调试；不直接用于对齐）
    global_cursor = 0

    # markers：暂存尚未分配到窗口里的标注事件
    markers: List[MarkerEvent] = []
    # marker_read_count：读到多少条 marker（包含非 annotation kind）
    marker_read_count = 0
    # marker_kept_count：真正保留下来的 annotation 数量
    marker_kept_count = 0

    # rhythm_stack：用栈维护“当前处于哪个节律区间”
    rhythm_stack: List[str] = []
    # last_watermark_global：从 marker 流拿到的“水位线”（表示 marker 已经至少送达到了哪个 global_sample）
    last_watermark_global: Optional[int] = None
    # session_meta：会话级元信息（比如 dataset_root / patient_id）
    session_meta: Dict[str, Any] = {}
    # current_source_record：当前来源 record（发送端在 segment_start 里告知）
    current_source_record: Optional[str] = None
    # current_source_segment_id：当前来源 segment id（发送端在 segment_start 里告知）
    current_source_segment_id: Optional[int] = None

    # WFDB 目录常用的 RECORDS 文件：列出所有 record_name
    records_path = out_dir / "RECORDS"
    # 如果已经存在旧的 RECORDS，先删除，避免混入旧记录
    if records_path.exists():
        records_path.unlink()

    # written：已写出窗口数量
    written = 0
    # last_data_time：上次收到 ECG 数据的时间（用于 session_end 后的“缓冲耗尽”判断）
    last_data_time = time.time()
    # session_end_global：收到 session_end marker 后记录它的 global_sample（用来触发退出）
    session_end_global: Optional[int] = None

    # verbose 模式下打印一些配置
    if args.verbose:
        print(f"[recv] fs={fs:g} window_samples={window_samples} watermark_delay_samples={delay_samples}")
        print(f"[recv] writing to: {out_dir}")

    try:
        # 主循环：不断拉取 ECG chunk 和 marker chunk，并在条件满足时写出窗口
        while True:
            # Pull ECG chunk (non-blocking)
            # 从 ECG inlet 拉取一个 chunk（timeout=0 表示不阻塞）
            chunk, _ts = inlet_ecg.pull_chunk(timeout=0.0, max_samples=4096)
            # chunk 非空表示收到了数据
            if chunk:
                # 转成 float32 的 1D numpy 数组
                arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
                # arr.size > 0 才加入缓冲
                if arr.size:
                    # 追加到缓冲区
                    ecg_buf.append(arr)
                    # 更新缓冲区长度（样本点数）
                    ecg_buf_len += int(arr.size)
                    # 更新累计游标（仅计数用途）
                    global_cursor += int(arr.size)
                    # 更新最近一次收到数据的时间
                    last_data_time = time.time()

            # Pull all available markers
            # 从 annotation inlet 拉取 marker chunk（同样非阻塞）
            m_chunk, _m_ts = inlet_ann.pull_chunk(timeout=0.0, max_samples=4096)
            # 如果有 marker 数据
            if m_chunk:
                # 逐条处理 marker（每个 m_sample 通常是一个 string 列表）
                for m_sample in m_chunk:
                    # 解析 marker；解析失败则跳过
                    ev = _parse_marker(m_sample)
                    if ev is None:
                        continue
                    # 统计读到的 marker 数量
                    marker_read_count += 1
                    # kind 表示 marker 类型（发送端约定的字段）
                    kind = str(ev.raw.get("kind", ""))
                    # ecg_chunk_end：发送端的“水位线”事件，表示 marker 送达进度（保证窗口末尾之前的标注已到齐）
                    if kind == "ecg_chunk_end":
                        last_watermark_global = (
                            ev.global_sample
                            if last_watermark_global is None
                            else max(last_watermark_global, ev.global_sample)
                        )
                        continue
                    # session_start：会话开始，携带一些元信息
                    if kind == "session_start":
                        # Keep basic context for later embedding into WFDB comments.
                        session_meta = {
                            "dataset_root": ev.raw.get("dataset_root", ""),
                            "patient_id": ev.raw.get("patient_id", ""),
                        }
                        continue
                    # segment_start：原始数据段开始（用于溯源），记录当前来源 record/segment_id
                    if kind == "segment_start":
                        current_source_record = str(ev.raw.get("record", "")) or current_source_record
                        try:
                            current_source_segment_id = int(ev.raw.get("segment_id"))
                        except Exception:
                            # 转换失败就保持原值不变
                            current_source_segment_id = current_source_segment_id
                        continue
                    # session_end：会话结束，记录结束的 global_sample
                    if kind == "session_end":
                        session_end_global = ev.global_sample
                        continue

                    # Keep only actual annotations (beat + rhythm tokens). Segment/session markers are ignored.
                    # 只保留真正的 annotation；其他 kind 已经在上面处理/忽略
                    if kind != "annotation":
                        continue
                    # 没有 symbol 的 annotation 没意义（不写）
                    if not ev.symbol:
                        continue
                    # 放入 markers 缓冲（后面按窗口切分）
                    markers.append(ev)
                    marker_kept_count += 1

            # If we have enough ECG to finalize a window AND the marker stream watermark
            # tells us that all markers up to the window end have been delivered.
            # 计算当前窗口的全局结束位置（注意：这里基于 buf_start_global + window_samples）
            window_end_global = buf_start_global + window_samples
            # watermark_target：窗口结束 + 额外延迟；水位线超过这个值才允许落盘
            watermark_target = window_end_global + delay_samples
            # 只要满足“缓冲够一个窗口”且“水位线足够”，就可以不断写窗口（可能一次循环写多个）
            while (
                ecg_buf_len >= window_samples
                and (last_watermark_global is not None)
                and (last_watermark_global >= watermark_target)
            ):
                # Materialize first window_samples from ecg_buf
                # 把 ecg_buf 的前 window_samples 个样本“实体化”成一个连续数组 window
                window_parts: List[np.ndarray] = []
                # remaining：还需要从缓冲里取多少样本点
                remaining = window_samples
                # 按 chunk 逐个取，直到凑够 window_samples
                while remaining > 0 and ecg_buf:
                    # 取队首 chunk
                    head = ecg_buf[0]
                    # 如果这个 chunk 整块都能放进窗口
                    if head.size <= remaining:
                        window_parts.append(head)
                        remaining -= int(head.size)
                        ecg_buf.pop(0)
                    else:
                        # chunk 比 remaining 长：切一段出来放进窗口，剩下的留在缓冲队首
                        window_parts.append(head[:remaining])
                        ecg_buf[0] = head[remaining:]
                        remaining = 0
                # 拼接成最终窗口数组
                window = np.concatenate(window_parts, axis=0)
                # 确认窗口长度正确（用于开发期自检）
                assert window.shape[0] == window_samples
                # 缓冲区长度扣掉已消费的 window_samples
                ecg_buf_len -= window_samples

                # 当前窗口的全局起点/终点（全局采样坐标）
                window_start_global = buf_start_global
                window_end_global = buf_start_global + window_samples

                # Extract markers for this window (keep list sorted-ish)
                # Note: markers list may not be strictly sorted due to transport; sort within a small bound.
                # 先按 global_sample 排序，确保按时间顺序处理（LSL 传输可能导致轻微乱序）
                markers.sort(key=lambda e: e.global_sample)
                # in_window：当前窗口内的 marker 列表
                in_window: List[MarkerEvent] = []
                # 丢掉（并用于更新节律栈）所有严格早于窗口起点的 marker
                while markers and markers[0].global_sample < window_start_global:
                    # Update rhythm state for anything strictly before the window.
                    ev0 = markers.pop(0)
                    if ev0.symbol == "+" and ev0.aux_note:
                        _update_rhythm_stack(rhythm_stack, ev0.aux_note)

                # Snapshot state at window start (for injection)
                # 记录窗口起点处的节律状态（用于 inject_rhythm_state）
                rhythm_state_at_start = list(rhythm_stack)

                # 把 [window_start_global, window_end_global) 区间内的 marker 取出来
                while markers and markers[0].global_sample < window_end_global:
                    ev0 = markers.pop(0)
                    in_window.append(ev0)
                    # 在把 marker 放入窗口的同时，更新节律栈（这样下一窗口的起点状态正确）
                    if ev0.symbol == "+" and ev0.aux_note:
                        _update_rhythm_stack(rhythm_stack, ev0.aux_note)

                # 生成 WFDB record_name（6 位递增编号）
                record_name = f"{args.record_prefix}_{written:06d}"
                # markers_meta：把 session_meta 复制一份，并补充“窗口起点对应的来源 record/segment”
                markers_meta: Dict[str, Any] = dict(session_meta)
                if current_source_record:
                    markers_meta["source_record_at_start"] = current_source_record
                if current_source_segment_id is not None:
                    markers_meta["source_segment_id_at_start"] = current_source_segment_id
                # 写出这一窗口的 WFDB 文件
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
                # 将 record_name 追加写入 RECORDS 文件（一行一个 record）
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(record_name + "\n")

                # 更新已写窗口计数
                written += 1
                # 窗口起点往前移动 hop_samples（这里 hop==window）
                buf_start_global += hop_samples
                # verbose：打印本窗口的起点、当前节律、标注数等
                if args.verbose:
                    active = rhythm_state_at_start[-1] if rhythm_state_at_start else "None"
                    print(
                        f"[write] {record_name} start={window_start_global} active_rhythm={normalize_rhythm_label(active) if active!='None' else 'None'} markers={len(in_window)}"
                    )

                # 达到 --max-windows 时提前退出
                if args.max_windows and written >= int(args.max_windows):
                    if args.verbose:
                        print("[stop] reached --max-windows")
                    return 0

            # Stop condition: if sender ended and we have written all full windows and no new data for a bit.
            # 如果收到 session_end，并且一段时间没有新数据且缓冲不够一个完整窗口，则认为已结束
            if session_end_global is not None:
                # If no new data recently and not enough buffered for another full window, stop.
                if (time.time() - last_data_time) > 2.0 and ecg_buf_len < window_samples:
                    if args.verbose:
                        print(
                            f"[stop] session_end received and buffers drained. markers_read={marker_read_count} markers_kept={marker_kept_count} watermark={last_watermark_global}"
                        )
                    break

            # 小睡一下，避免 while True 忙等占满 CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        # Ctrl+C 手动中断时走这里（verbose 会打印提示）
        if args.verbose:
            print("[stop] KeyboardInterrupt")

    # 正常结束返回 0
    return 0


# 作为脚本直接运行时，从 main() 退出并把返回码交给系统
if __name__ == "__main__":
    raise SystemExit(main())
