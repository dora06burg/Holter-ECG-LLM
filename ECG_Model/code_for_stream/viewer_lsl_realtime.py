# 这个脚本做什么？
# - 从两条 LSL（Lab Streaming Layer）流实时接收数据：
#   1) ECG 连续波形流（默认名：icentia_ecg，float32，1 通道）
#   2) 标注/事件流（默认名：icentia_ann，string，JSON marker）
# - 维护一个“滚动显示窗口”（display-sec），把最近一段 ECG 画出来
# - 同时把 beat 标注（symbol != '+') 画成散点，把节律 token（symbol == '+' 且 aux_note 有值）解析成节律状态
# - 如果使用 --no-plot，则不打开 matplotlib GUI，只在终端每秒打印一次统计信息
#
# 快速上手（示例）：
# - 先启动发送端（例如 sender_icentia_lsl.py）
# - 再运行本脚本：
#   python viewer_lsl_realtime.py --display-sec 12 --refresh-hz 10
# - 若你在无 GUI 环境（服务器/WSL 无显示），可用：
#   python viewer_lsl_realtime.py --no-plot

# 未来导入：让类型注解在运行时不立即求值（常用于前向引用/降低开销）
from __future__ import annotations

# argparse：解析命令行参数（LSL 流名、显示时长等）
import argparse
# json：解析 marker 流里的 JSON 字符串
import json
# time：用于刷新/限时退出/sleep 控制循环频率
import time
# deque：固定长度的双端队列，用来做“滚动缓冲区”（自动丢弃最旧样本）
from collections import deque
# dataclass：定义简单的数据结构（BeatMarker/RhythmToken）
from dataclasses import dataclass
# typing：类型标注，提高可读性与 IDE 体验
from typing import Any, Deque, Dict, List, Optional, Tuple

# numpy：用于把 deque 转成数组、生成 x 轴、拼散点坐标等
import numpy as np

# icentia_wfdb：本项目内的辅助函数（节律标签归一化、aux_note 清洗）
from icentia_wfdb import normalize_rhythm_label, safe_aux_note
# lsl_utils：延迟导入 pylsl（缺依赖时给更友好提示）
from lsl_utils import import_pylsl


# BeatMarker：用于保存“搏动/心拍”标注（例如 N、V 等），在图上画点
@dataclass
class BeatMarker:
    # global_sample：全局采样点索引（与发送端的 --fs-out 时间轴对齐）
    global_sample: int
    # symbol：WFDB 的 beat symbol（这里把 symbol != '+' 的都当 beat 来显示）
    symbol: str


# RhythmToken：用于保存“节律 token”（symbol='+'，aux_note 里是 '(AFIB'、')' 等）
@dataclass
class RhythmToken:
    # global_sample：token 发生的全局采样点
    global_sample: int
    # aux_note：WFDB aux_note（携带节律开始/结束信息）
    aux_note: str


# 构建命令行参数解析器
def _build_argparser() -> argparse.ArgumentParser:
    # 创建 parser 并写明脚本用途
    p = argparse.ArgumentParser(description="Realtime viewer for Icentia LSL ECG + markers.")
    # ECG 流的 LSL name（需和发送端一致）
    p.add_argument("--lsl-name-ecg", type=str, default="icentia_ecg")
    # ANN/marker 流的 LSL name（需和发送端一致）
    p.add_argument("--lsl-name-ann", type=str, default="icentia_ann")
    # resolve 超时：找不到对应 name 的流时最多等多久（秒）
    p.add_argument("--resolve-timeout", type=float, default=10.0)

    # display-sec：滚动窗口显示最近多少秒的 ECG
    p.add_argument("--display-sec", type=float, default=12.0, help="How many seconds to display in the rolling window.")
    # refresh-hz：绘图刷新频率（越高越流畅，但 CPU 越高）
    p.add_argument("--refresh-hz", type=float, default=10.0, help="Plot refresh rate.")
    # fs：手动覆盖采样率（0 表示从 LSL 流的 nominal_srate 读取）
    p.add_argument("--fs", type=float, default=0.0, help="Override sampling rate (0 = read from stream info).")
    # duration-sec：运行多少秒后自动退出（0 表示直到 Ctrl-C）
    p.add_argument("--duration-sec", type=float, default=0.0, help="Run for N seconds then exit (0 = run until Ctrl-C).")
    # no-plot：不启用 matplotlib GUI，改为打印统计信息（适合无显示环境）
    p.add_argument("--no-plot", action="store_true", help="Disable matplotlib GUI; print streaming stats to stdout.")
    # 返回 parser
    return p


# 根据 LSL stream name 解析出一个 StreamInfo（用于创建 StreamInlet）
def _resolve_one_stream(pylsl, *, name: str, timeout: float):
    # 按属性 name 在 LSL 网络中查找流
    results = pylsl.resolve_byprop("name", name, timeout=timeout)
    # 找不到就退出
    if not results:
        raise SystemExit(f"Could not resolve LSL stream with name='{name}' (timeout={timeout}s)")
    # 只取第一个匹配结果（通常同名只有一个）
    return results[0]


# 从 ANN 流的一条样本（字符串列表）解析出 marker 的 dict
def _parse_marker(sample: List[str]) -> Optional[Dict[str, Any]]:
    # 空样本直接忽略
    if not sample:
        return None
    # sample[0] 期望是 JSON 字符串
    try:
        raw = json.loads(sample[0])
    except Exception:
        # JSON 解析失败则忽略
        return None
    # 只接受 dict（JSON object）
    if not isinstance(raw, dict):
        return None
    # 返回 marker 原始 dict（后续根据 kind/symbol/aux_note 处理）
    return raw


# 程序入口：连接 LSL，接收数据，实时画图/打印
def main() -> int:
    # 解析命令行参数
    args = _build_argparser().parse_args()
    # 动态导入 pylsl（LSL Python SDK）
    pylsl = import_pylsl()

    # resolve 两条流：ECG + ANN
    stream_ecg = _resolve_one_stream(pylsl, name=str(args.lsl_name_ecg), timeout=float(args.resolve_timeout))
    stream_ann = _resolve_one_stream(pylsl, name=str(args.lsl_name_ann), timeout=float(args.resolve_timeout))
    # 创建 inlet：接收端句柄；max_buflen 表示 LSL 内部最多缓存多少秒的数据
    inlet_ecg = pylsl.StreamInlet(stream_ecg, max_buflen=60)
    inlet_ann = pylsl.StreamInlet(stream_ann, max_buflen=60)

    # 采样率：优先用命令行 --fs 覆盖，否则从 ECG 流的 nominal_srate 读取
    fs = float(args.fs) if float(args.fs) > 0 else float(inlet_ecg.info().nominal_srate())
    # 采样率必须为正；否则无法把样本点转换成“秒”来显示
    if fs <= 0:
        raise SystemExit("ECG stream nominal_srate is not set. Use --fs to override.")

    # 根据 display-sec 计算滚动窗口长度（采样点数）
    display_samples = int(round(float(args.display_sec) * fs))
    # 至少保留 10 个点，避免窗口太小导致绘图/逻辑异常
    display_samples = max(10, display_samples)

    # ecg：滚动缓冲区（只保留最近 display_samples 个样本）
    ecg: Deque[float] = deque(maxlen=display_samples)
    # beats：缓存 beat 标注（用于绘图的散点）
    beats: List[BeatMarker] = []
    # rhythm_tokens：缓存节律 token（用于在窗口附近显示/调试）
    rhythm_tokens: List[RhythmToken] = []

    # global_cursor：累计接收的 ECG 样本数（全局采样点索引；与发送端 global_sample 对齐的“同一时间轴”）
    global_cursor = 0
    # rhythm_stack：当前“节律区间栈”，栈顶表示当前激活节律（例如 AFIB）
    rhythm_stack: List[str] = []
    # last_segment：最近一次收到 segment_start 时记录的 record 名（用于显示）
    last_segment: Optional[str] = None

    # Plot (optional)
    # 这些变量在 --no-plot 时保持 None；在 plot 模式下会被初始化为 matplotlib 对象
    fig = ax = line = beat_scatter = txt = None
    if not args.no_plot:
        # 仅在需要绘图时才导入 matplotlib（避免无 GUI 环境导入失败）
        import matplotlib.pyplot as plt

        # 打开交互模式：允许循环里不断刷新图像
        plt.ion()
        # 创建 figure + axes；设置一个较宽的画布
        fig, ax = plt.subplots(figsize=(10, 4))
        # line：ECG 曲线（先空数据，后续 update_plot 里 set_data）
        (line,) = ax.plot([], [], lw=1)
        # beat_scatter：beat 的散点（先空数据，后续更新 offsets）
        beat_scatter = ax.scatter([], [], s=20)
        # txt：左上角文本框，显示采样率/全局索引/节律等状态
        txt = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left")
        # 坐标轴/标题文字
        ax.set_xlabel("Time (s) in buffer")
        ax.set_ylabel("ECG (mV)")
        ax.set_title("LSL ECG stream (rolling)")

    # last_refresh：上一次刷新绘图的时间戳（秒）
    last_refresh = 0.0
    # refresh_period：每次刷新间隔（秒）= 1/refresh_hz（做下限避免除零）
    refresh_period = 1.0 / max(1e-6, float(args.refresh_hz))
    # t_rate：用于估算“有效采样率”的统计窗口起点时间
    t_rate = time.time()
    # samples_in_last_sec：在统计窗口内累计收到多少样本
    samples_in_last_sec = 0
    # eff_rate：估算的“接收有效采样率”（Hz），用于显示/打印
    eff_rate = 0.0

    # update_plot：把当前缓冲区和 marker 状态画到图上
    def update_plot():
        # nonlocal：我们会在函数内部更新外层的 beats/rhythm_tokens/eff_rate
        nonlocal beats, rhythm_tokens, eff_rate

        # 把 deque 转成 numpy 数组（y 轴）
        y = np.asarray(ecg, dtype=np.float32)
        # 没有数据就不更新图
        if y.size == 0:
            return
        # x 轴：按采样率把样本点索引转换成“秒”（在缓冲区内的相对时间）
        x = np.arange(y.size, dtype=np.float32) / float(fs)
        # 静态类型/防御：确保 matplotlib 对象已初始化
        assert ax is not None and line is not None and beat_scatter is not None and txt is not None and fig is not None
        # 更新 ECG 曲线数据
        line.set_data(x, y)
        # x 轴范围设置为缓冲区覆盖的时间范围
        ax.set_xlim(float(x[0]), float(x[-1]))

        # Markers within buffer
        # buf_start_global：当前缓冲区起点在全局采样轴上的位置
        # 解释：global_cursor 指向“已收到样本的末尾”，减去缓冲长度就是缓冲起点
        buf_start_global = global_cursor - y.size

        # 丢弃早于缓冲区起点的 beat（避免列表无限增长）
        beats = [b for b in beats if b.global_sample >= buf_start_global]
        # rx/ry：beat 散点的坐标列表（时间秒、幅值 mV）
        rx = []
        ry = []
        # 遍历 beat，计算其在缓冲区内的相对位置
        for b in beats:
            rel = b.global_sample - buf_start_global
            if 0 <= rel < y.size:
                # x 坐标：相对采样点/采样率
                rx.append(float(rel) / float(fs))
                # y 坐标：取该采样点的波形值
                ry.append(float(y[int(rel)]))
        # 更新散点；若无点则传一个空数组（shape=(0,2)）
        beat_scatter.set_offsets(np.column_stack([rx, ry]) if rx else np.empty((0, 2)))

        # 节律 token 也做裁剪：保留缓冲区前 5 秒左右的 token（便于显示连续性/调试）
        rhythm_tokens = [r for r in rhythm_tokens if r.global_sample >= buf_start_global - int(5 * fs)]

        # 当前激活节律：取栈顶（若栈空则表示 None）
        active = rhythm_stack[-1] if rhythm_stack else ""
        # normalize：把原始标签映射到统一形式（例如 AFIB/NSR/AFL 等）
        active_norm = normalize_rhythm_label(active) if active else "None"
        # 更新文本框内容（包含采样率、有效接收率、全局样本计数、segment、节律、缓冲内 beat 数）
        txt.set_text(
            f"fs={fs:g}Hz  eff_rate~{eff_rate:0.1f}Hz  global_sample={global_cursor}\n"
            f"segment={last_segment or '-'}  rhythm={active_norm}  beats_in_buf={len(beats)}"
        )

        # Color by active rhythm (current)
        # 根据当前节律给背景上色：NSR=淡绿，AFIB=淡红，AFL=淡黄，其它=白色
        bg = {"NSR": "#F3FFF3", "AFIB": "#FFF3F3", "AFL": "#FFFBE6"}.get(active_norm, "#FFFFFF")
        ax.set_facecolor(bg)

        # 请求 GUI 刷新（非阻塞）
        fig.canvas.draw_idle()
        # 强制处理 GUI 事件队列，让图像真正更新
        fig.canvas.flush_events()

    # 记录主循环开始时间（用于 --duration-sec 退出）
    t_start = time.time()
    # last_print：上一次打印 stdout 的时间（仅 --no-plot 使用）
    last_print = 0.0
    # 主循环：不停拉取 ECG chunk 和 marker，并刷新/打印
    while True:
        # Pull ECG data
        # 从 ECG inlet 拉一批样本（chunk）；timeout=0 表示非阻塞
        chunk, _ts = inlet_ecg.pull_chunk(timeout=0.0, max_samples=4096)
        if chunk:
            # 转成一维 float32 数组
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            # 逐样本 append 到 deque（保持滚动窗口）
            for v in arr.tolist():
                ecg.append(float(v))
            # 更新全局采样计数
            global_cursor += int(arr.size)
            # 用于有效采样率统计
            samples_in_last_sec += int(arr.size)

        # Pull markers
        # marker 流是事件流；这里用 pull_sample 在同一轮里尽可能把“当前可用的 marker”全部取空
        while True:
            # 非阻塞取一个 marker（string list）
            m_sample, _m_ts = inlet_ann.pull_sample(timeout=0.0)
            if not m_sample:
                # 当前没有更多 marker 就退出内层循环
                break
            # 解析 JSON marker
            raw = _parse_marker(m_sample)
            if raw is None:
                continue
            # kind：事件类型（例如 segment_start/annotation/ecg_chunk_end 等）
            kind = str(raw.get("kind", ""))
            if kind == "segment_start":
                # 记录当前 segment 的 record 名，用于显示
                last_segment = str(raw.get("record", "")) or last_segment

            # global_sample：事件发生的全局采样点（用于与 ECG 对齐）
            gs = raw.get("global_sample")
            try:
                gs_i = int(gs)
            except Exception:
                continue

            # symbol：WFDB annotation 的 symbol（beat 或 '+' token）
            symbol = str(raw.get("symbol", ""))
            # aux_note：清洗后的 aux_note（可能为 '(AFIB' 或 ')' 等）
            aux = safe_aux_note(raw.get("aux_note"))

            # beat：symbol 非空且不等于 '+'（例如 'N'、'V' 等）
            if symbol and symbol != "+":
                beats.append(BeatMarker(global_sample=gs_i, symbol=symbol))
            # rhythm token：symbol='+' 且 aux_note 非空
            if symbol == "+" and aux:
                rhythm_tokens.append(RhythmToken(global_sample=gs_i, aux_note=aux))
                # Update rhythm state machine
                # 下面是一个简单的“节律状态机”：
                # - aux 以 '(' 开头且不以 ')' 结尾：进入某节律区间（例如 '(AFIB'）
                # - aux == ')'：退出最近的节律区间
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
        # 每隔约 1 秒估算一次“实际接收采样率”
        now = time.time()
        if now - t_rate >= 1.0:
            eff_rate = float(samples_in_last_sec) / float(now - t_rate)
            t_rate = now
            samples_in_last_sec = 0

        if args.no_plot:
            # no-plot 模式：每秒打印一次状态
            if now - last_print >= 1.0:
                active = rhythm_stack[-1] if rhythm_stack else ""
                active_norm = normalize_rhythm_label(active) if active else "None"
                print(
                    f"fs={fs:g}Hz eff_rate~{eff_rate:0.1f}Hz global_sample={global_cursor} "
                    f"segment={last_segment or '-'} rhythm={active_norm} beats_cached={len(beats)}"
                )
                last_print = now
        else:
            # plot 模式：按 refresh_period 刷新图像
            if now - last_refresh >= refresh_period:
                update_plot()
                last_refresh = now

        # duration-sec：达到运行时长则退出
        if float(args.duration_sec) > 0 and (now - t_start) >= float(args.duration_sec):
            break

        # 小睡一下避免忙等（占用过多 CPU）
        time.sleep(0.005)

    # 正常退出返回 0
    return 0


# 以脚本方式运行时执行 main() 并将其返回码作为进程退出码
if __name__ == "__main__":
    raise SystemExit(main())
