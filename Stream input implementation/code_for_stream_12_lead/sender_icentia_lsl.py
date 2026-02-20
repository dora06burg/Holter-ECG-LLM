# 把 Icentia11k 的 WFDB 记录“像实时采集一样”通过 LSL（Lab Streaming Layer）发出去。
#
# 你会得到两条 LSL 流：
# 1) ECG 流（float32，12 通道，名义采样率 = --fs-out）：连续波形样本
# 2) ANN 流（string，事件流，采样率=0）：每条是 JSON 字符串，描述注释/分段/水位线等事件
#
# 为什么要两条流？
# - ECG 是高频连续数据（例如 250Hz），适合用 push_chunk 连续发送
# - 注释/事件是低频离散数据（节律标签、搏动符号、段落开始/结束），适合用 marker 事件发送
#
# 快速上手（典型用法）：
# - 先在另一个终端启动接收端（例如 receiver/viewer 脚本），确保能连接到 LSL
# - 再运行本脚本，例如：
#   python sender_icentia_lsl.py --dataset ../dataset --patient 0 --segment-start 0 --segment-end 49 --fs-out 250 --chunk-ms 200 --speed 1
#
# 关键概念：
# - global_sample：全局输出采样点索引（以 --fs-out 为基准），跨 segment 单调递增，便于接收端对齐窗口
# - ecg_chunk_end：水位线 marker，告诉接收端“某个 chunk 已结束”，高倍速/不等时读取时很有用

from __future__ import annotations  # 允许在类型注解中使用前向引用（注解字符串化），减少类型导入问题

import argparse  # 解析命令行参数
import json  # 将 marker 字典序列化为 JSON 字符串（ANN 流发送的是字符串）
import os  # 处理默认路径等
import sys  # 调整模块搜索路径
import time  # 控制“仿实时”的播放节奏（sleep）以及超时等待
from dataclasses import dataclass  # 用 dataclass 简化配置对象定义
from pathlib import Path  # 统一处理路径
from typing import Dict, List, Optional  # 类型标注：更易读/更好提示

import numpy as np  # 数值运算；本脚本主要用来处理采样数组/reshape 等
# 允许从原始 code_for_stream 目录复用工具模块
_THIS_DIR = Path(__file__).resolve().parent
_ORIG_DIR = _THIS_DIR.parent / "code_for_stream"
if _ORIG_DIR.exists() and str(_ORIG_DIR) not in sys.path:
    sys.path.insert(0, str(_ORIG_DIR))
# 工具函数（与 WFDB/Icentia11k 数据集读写、重采样、标签归一化相关）
from icentia_wfdb import (
    list_patient_records,  # 列出某个病人（pXXXXX）拥有的所有 segment 记录（record id 列表）
    normalize_beat_symbol,  # 规范化搏动符号（例如将一些符号映射到统一集合）
    normalize_rhythm_label,  # 规范化节律标签（aux_note 中的 (AFIB 等)）
    parse_record_id,  # 解析 record 字符串到结构化信息（本文件里未直接用到，保留为工具）
    read_annotations,  # 读取 WFDB 注释（atr）得到 sample/symbol/aux_note
    read_ecg_mV,  # 读取 WFDB 波形（dat+hea），并转换为 mV（毫伏）单位的 numpy 数组
    read_segment_header,  # 读取 WFDB 头信息（hea）得到 fs/sig_len 等元数据
    resample_samples,  # 把“注释的采样点索引”从 fs_in 映射到 fs_out（只重映射索引，不改变含义）
    resample_signal_polyphase,  # 把波形从 fs_in 真正重采样到 fs_out（polyphase/抗混叠）
    resolve_dataset_root,  # 允许你传 .../dataset 或 .../dataset/Icentia11k，统一解析到真正根目录
    safe_aux_note,  # 让 aux_note 更安全可用（例如处理 None、字节串、奇怪字符等）
)
# 这个库不能正常下载，因此用这种方式直接引入（lsl_utils 里通常做了“优雅导入/提示安装”的封装）
from lsl_utils import import_pylsl  # 动态导入 pylsl（LSL 的 Python SDK）


@dataclass(frozen=True)
class StreamConfig:
    name_ecg: str  # ECG 流的 LSL name（接收端通常按 name/type 发现）
    name_ann: str  # ANN 流的 LSL name（发送 JSON marker）
    type_ecg: str = "ECG"  # LSL type：用于分类（不是强制，但很常用）
    type_ann: str = "Markers"  # LSL type：事件/标记流


# 统一导联顺序（全项目唯一标准）
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ICENTIA_LEAD_MASK = [1] + [0] * (len(LEAD_ORDER) - 1)
ICENTIA_SOURCE_LEADS = ["modified_I"]


def adapt_icentia11k_to_12lead(x_mV: np.ndarray) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Icentia11k (CardioSTAT) 单导联 -> 标准 12 导联格式。
    - modified lead I 映射到 Lead I
    - 其余导联补 0
    """
    n = int(x_mV.shape[0])
    signal_12 = np.zeros((n, len(LEAD_ORDER)), dtype=np.float32)
    signal_12[:, 0] = x_mV.astype(np.float32, copy=False)
    lead_mask = list(ICENTIA_LEAD_MASK)
    source_leads = list(ICENTIA_SOURCE_LEADS)
    return signal_12, lead_mask, source_leads


def _build_argparser() -> argparse.ArgumentParser:
    # 创建参数解析器；把所有可调参数集中在这里，便于命令行使用和查看 --help
    p = argparse.ArgumentParser(  # argparse.ArgumentParser：Python 标准库的命令行解析器
        description="Stream Icentia11k WFDB records as real-time LSL (ECG + annotation markers)."  # 脚本用途简介
    )
    # 数据集根目录：既可以指向 .../dataset，也可以指向 .../dataset/Icentia11k
    p.add_argument("--dataset", type=str, default=os.path.join("..", "dataset"), help="Dataset root (either .../dataset or .../dataset/Icentia11k).")
    # 病人 id（例如 patient=0 对应 p00000）；Icentia11k 的文件通常以 pxxxxx_syy 表示
    p.add_argument("--patient", type=int, default=0, help="Patient id (e.g., 0 for p00000).")

    # segment streaming 范围（包含端点）；用于只播放某些片段而不是整个病人所有片段
    p.add_argument("--segment-start", type=int, default=0, help="First segment id to stream (inclusive).")
    p.add_argument("--segment-end", type=int, default=49, help="Last segment id to stream (inclusive).")

    # 输出采样率：ECG 会真正重采样到这个频率；注释采样点索引也会被映射到这个频率的时间轴
    p.add_argument("--fs-out", type=float, default=250.0, help="Output sampling rate (Hz). True resampling will be applied if different from input.")
    # 每次 push 到 LSL 的 ECG chunk 时长（毫秒）；chunk 越小延迟越低，但系统开销越大
    p.add_argument("--chunk-ms", type=float, default=200.0, help="ECG push chunk size (milliseconds). Smaller = lower latency, higher overhead.")
    # 播放速度：1.0=实时；10.0=10 倍速；0=不 sleep，尽可能快地发完（离线回放/压力测试）
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed. 1.0 = real-time. 10.0 = 10x faster. 0 = as fast as possible.")
    # watermark（水位线）频率：每 N 个 ECG chunk 发送一次 kind=ecg_chunk_end 的 marker
    # 接收端可以用它来“确认窗口切分点”，尤其在高倍速或接收端拉取不稳定时更可靠
    p.add_argument(
        "--watermark-every-chunks",
        type=int,
        default=1,
        help="Send an ANN watermark marker (`kind=ecg_chunk_end`) every N ECG chunks (helps receivers finalize windows correctly even at high speed).",
    )

    # LSL 流名：接收端通常用 name/type 来 resolve（发现）对应流
    p.add_argument("--lsl-name-ecg", type=str, default="icentia_ecg", help="LSL stream name for ECG samples.")
    p.add_argument("--lsl-name-ann", type=str, default="icentia_ann", help="LSL stream name for annotation markers (JSON strings).")
    # source_id：LSL 的唯一来源标识（可选）；留空则脚本会自动生成（包含 patient_id 和 fs_out）
    p.add_argument("--source-id", type=str, default="", help="LSL source_id (optional).")
    # 是否等待消费者连接：避免你用高倍速播放时，接收端还没启动就错过开头的样本
    p.add_argument(
        "--wait-for-consumers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait until both ECG+ANN outlets have at least one consumer before streaming (prevents missing the beginning at high --speed).",
    )
    # 等待消费者连接的超时时间（秒）
    p.add_argument(
        "--wait-timeout-sec",
        type=float,
        default=30.0,
        help="Max seconds to wait for LSL consumers when --wait-for-consumers is enabled.",
    )

    # 最多发送多少“输出采样率下”的秒数；用于只回放前几秒做调试（0 表示无限制）
    p.add_argument("--max-seconds", type=float, default=0.0, help="Stop after streaming this many seconds of *output* ECG (0 = no limit).")
    # verbose：打印更多调试信息（当前 segment、警告、停止原因等）
    p.add_argument("--verbose", action="store_true")
    return p  # 返回构建好的解析器


def _lsl_outlets(pylsl, cfg: StreamConfig, *, fs_out: float, source_id: str, patient_id: int, dataset_root: str):
    # 如果用户没有显式指定 source_id，就生成一个稳定的默认值（同一配置下更容易被接收端区分/缓存）
    if not source_id:
        source_id = f"icentia11k_p{patient_id:05d}_fs{fs_out:g}"
    # StreamInfo：LSL 流的“元数据/身份证 + 说明书”
    # 说明：LSL 的发现机制会广播 StreamInfo；接收端 resolve 时读取这些信息以正确解释数据。
    #
    # 接收端通常关心：
    # - 这条流的 name / type（是不是 ECG / Markers）
    # - channel_count（几个通道）
    # - channel_format（float32/string 等）
    # - nominal_srate（名义采样率；对 ECG 很关键）
    # 从而能把接收到的字节/数组解释为正确的浮点 ECG 流或字符串 marker 流。
    info_ecg = pylsl.StreamInfo(  # 创建 ECG 流的 StreamInfo
        name=cfg.name_ecg,  # LSL name（默认 icentia_ecg）
        type=cfg.type_ecg,  # LSL type（默认 ECG）
        channel_count=len(LEAD_ORDER),  # 统一为 12 导联
        nominal_srate=float(fs_out),  # 名义采样率：告诉接收端这是 250Hz 还是其他
        channel_format="float32",  # 数据格式：float32（更通用、带宽适中）
        source_id=source_id + "_ecg",  # source_id：同一设备/源的唯一标识（这里加后缀区分 ecg/ann）
    )
    # 给 ECG 流加一些额外描述（XML 结构），方便调试/在接收端读取元信息
    chs = info_ecg.desc().append_child("channels")  # 添加 channels 节点
    for lead in LEAD_ORDER:
        ch = chs.append_child("channel")
        ch.append_child_value("label", lead)
        ch.append_child_value("unit", "mV")
    info_ecg.desc().append_child_value("dataset_root", dataset_root)  # 记录数据集根路径（便于追溯）
    info_ecg.desc().append_child_value("patient_id", f"p{patient_id:05d}")  # 记录病人 id（p00000）
    info_ecg.desc().append_child_value("fs_out", str(fs_out))  # 记录输出采样率（字符串形式）

    # 创建 ANN（marker）流的 StreamInfo：它是事件流，不是连续采样，所以 nominal_srate = 0
    info_ann = pylsl.StreamInfo(
        name=cfg.name_ann,  # LSL name（默认 icentia_ann）
        type=cfg.type_ann,  # LSL type（默认 Markers）
        channel_count=1,  # 每条 marker 只发一个字符串（JSON）
        nominal_srate=0.0,  # 事件流一般设置为 0（不表示固定采样率）
        channel_format="string",  # 事件内容用字符串传（这里是 JSON）
        source_id=source_id + "_ann",  # source_id 后缀 _ann
    )
    # 同样附加一些元信息到 ANN 流
    info_ann.desc().append_child_value("dataset_root", dataset_root)
    info_ann.desc().append_child_value("patient_id", f"p{patient_id:05d}")
    info_ann.desc().append_child_value("fs_out", str(fs_out))
    info_ann.desc().append_child_value("format", "json")  # 明确约定：ANN 流 payload 是 JSON

    # StreamOutlet：可以理解为“发射器/发送端”
    # 本质上它封装了：发送缓冲区 + 网络广播/发现 + 对外 push_sample/push_chunk 的 API。
    # 一旦创建 outlet，LSL 网络里就“出现”了一条可被发现的流（携带刚才的 StreamInfo）。
    outlet_ecg = pylsl.StreamOutlet(info_ecg, chunk_size=0, max_buffered=60)  # ECG outlet（缓冲最多约 60 秒）
    outlet_ann = pylsl.StreamOutlet(info_ann, chunk_size=0, max_buffered=60)  # ANN outlet
    return outlet_ecg, outlet_ann  # 返回两个 outlet 供 main() 使用


def _push_marker(outlet_ann, marker: Dict):
    # 把 marker 字典转成 JSON 字符串，并作为“单通道字符串样本”发到 ANN 流
    # ensure_ascii=False：保证中文/特殊字符不被转义，接收端更易读
    outlet_ann.push_sample([json.dumps(marker, ensure_ascii=False)])  # LSL string 流的 sample 是 list[str]（长度=channel_count=1）


def main() -> int:
    # 解析命令行参数（args.* 就是你在命令行里传入的配置）
    args = _build_argparser().parse_args()
    # 动态导入 pylsl（LSL Python SDK）；如果环境没装好，这里会抛出更友好的提示
    pylsl = import_pylsl()

    # 解析数据集根目录：允许传 dataset 或 dataset/Icentia11k，最终统一到 Icentia11k 的根目录
    dataset_root = resolve_dataset_root(args.dataset)
    # 病人 id（转换为 int，确保后面格式化成 p00000）
    patient_id = int(args.patient)
    # 找到该病人的所有 record（每个 record 对应一个 segment）
    patient_records = list_patient_records(dataset_root, patient_id)
    # 如果没有记录，直接退出（通常是路径不对或 patient id 不存在）
    if not patient_records:
        raise SystemExit(f"No records found for patient p{patient_id:05d} under: {dataset_root}")

    # 根据 segment 范围过滤（只播放选定片段）
    patient_records = [r for r in patient_records if args.segment_start <= r.segment_id <= args.segment_end]
    # 过滤后为空就退出
    if not patient_records:
        raise SystemExit("No records after applying --segment-start/--segment-end.")

    # 读取第一个 segment 的头信息，作为输入采样率的参考值（理论上同一个病人 segments 采样率应一致）
    first_seg = read_segment_header(dataset_root, patient_records[0])
    fs_in_ref = float(first_seg.fs)  # 输入采样率（Hz）

    # 输出采样率（Hz）：后面会重采样到这个频率
    fs_out = float(args.fs_out)
    # 基本参数校验：采样率必须 > 0
    if fs_out <= 0:
        raise SystemExit("--fs-out must be > 0.")

    # 构建 LSL 流配置（name/type 等）
    cfg = StreamConfig(name_ecg=args.lsl_name_ecg, name_ann=args.lsl_name_ann)
    # LSL 的核心思想（简化理解）：
    # - StreamInfo：描述流是什么（元数据）
    # - StreamOutlet：发送端（创建后可被发现）
    # - push_sample/push_chunk：把数据塞进发送缓冲并广播出去
    # - 接收端（StreamInlet）按 name/type/source_id resolve 到流，再持续拉取数据
    # 这里创建两条流：一条发 ECG 波形，一条发注释/事件（marker）
    outlet_ecg, outlet_ann = _lsl_outlets(
        pylsl,  # pylsl 模块（包含 StreamInfo/StreamOutlet 等）
        cfg,  # 流名字/类型配置
        fs_out=fs_out,  # 输出采样率（写入 StreamInfo）
        source_id=str(args.source_id),  # 允许用户自定义 source_id
        patient_id=patient_id,  # 写入元信息
        dataset_root=dataset_root,  # 写入元信息
    )

    # 把 chunk-ms（毫秒）换算成每个 chunk 的采样点数
    chunk_samples = int(round(fs_out * (float(args.chunk_ms) / 1000.0)))
    # 至少要 1 个点，否则 push_chunk 会无意义/出错
    chunk_samples = max(1, chunk_samples)
    # watermark 频率至少为 1（每个 chunk 都发或每 N 个 chunk 发）
    watermark_every = max(1, int(args.watermark_every_chunks))

    # 可选：等待接收端连接上来（两个 outlet 都有消费者）再开始发送
    if args.wait_for_consumers:
        # verbose 模式下打印提示
        if args.verbose:
            print("[wait] waiting for LSL consumers (ECG + ANN)...")
        t0 = time.time()  # 记录开始等待的时间
        while True:
            # have_consumers()：LSL outlet 侧查询当前是否至少有一个 inlet 连接/订阅
            if outlet_ecg.have_consumers() and outlet_ann.have_consumers():
                if args.verbose:
                    print("[wait] consumer connected, starting stream.")
                break  # 两条流都有人接收了，开始正式 streaming
            # 超时检查：避免永远卡住
            if (time.time() - t0) > float(args.wait_timeout_sec):
                raise SystemExit(
                    "Timed out waiting for LSL consumers. "
                    "Start the receiver/viewer first, or run with --no-wait-for-consumers."
                )
            time.sleep(0.05)  # 小睡一会儿再轮询，避免 CPU 空转

    global_sample_offset = 0  # 全局输出采样点偏移（跨 segment 累加）；用于构造 global_sample
    # 把 max-seconds（秒）换算成“输出域”的最大样本数；0 表示无限制
    max_samples = int(round(float(args.max_seconds) * fs_out)) if args.max_seconds and args.max_seconds > 0 else 0

    _push_marker(
        outlet_ann,
        {
            "kind": "session_start",  # marker 类型：会话开始
            "global_sample": 0,  # 全局采样点从 0 开始计
            "patient_id": f"p{patient_id:05d}",  # 病人 id（字符串）
            "dataset_root": dataset_root,  # 数据集根路径（便于接收端日志/复现）
            "fs_in": fs_in_ref,  # 输入采样率参考值
            "fs_out": fs_out,  # 输出采样率（真正发送的 ECG 采样率）
            "chunk_samples": chunk_samples,  # 每个 chunk 的采样点数（接收端可以用来切窗）
            "speed": float(args.speed),  # 播放速度（用于接收端记录）
            "segment_start": int(args.segment_start),  # 会话播放的 segment 起点
            "segment_end": int(args.segment_end),  # 会话播放的 segment 终点
            "lead_order": list(LEAD_ORDER),
            "lead_mask": list(ICENTIA_LEAD_MASK),
            "source_leads": list(ICENTIA_SOURCE_LEADS),
            "source_device": "CardioSTAT",
        },
    )

    t_start = time.time()  # 记录会话开始的墙钟时间（wall clock），用于统计 elapsed
    streamed_samples = 0  # 累计已发送的输出样本数（用于 --max-seconds 限制）

    # 逐个 segment（record）回放
    for rec in patient_records:
        seg = read_segment_header(dataset_root, rec)  # 读本 segment 的头信息（长度/采样率等）
        fs_in = float(seg.fs)  # 该 segment 的输入采样率（Hz）
        # 如果 segment 的采样率与参考值不一致，给个警告（一般不影响运行，但可能说明数据异常）
        if abs(fs_in - fs_in_ref) > 1e-6 and args.verbose:
            print(f"[warn] segment {rec.record} fs differs: {fs_in} vs {fs_in_ref}")

        # verbose 打印当前 segment 基本信息
        if args.verbose:
            print(f"[seg] start {rec.record} (fs_in={fs_in}, fs_out={fs_out}) global_offset={global_sample_offset}")

        _push_marker(
            outlet_ann,
            {
                "kind": "segment_start",  # marker 类型：segment 开始
                "global_sample": int(global_sample_offset),  # 当前 segment 的起始 global_sample
                "record": rec.record,  # record 名（例如 p00000_s01）
                "segment_id": int(rec.segment_id),  # segment 编号（例如 1）
                "fs_in": fs_in,  # 输入采样率（用于接收端记录）
                "fs_out": fs_out,  # 输出采样率（用于接收端记录）
                "sig_len_in": int(seg.sig_len),  # 输入域的样本数（原始长度）
                "lead_order": list(LEAD_ORDER),
                "lead_mask": list(ICENTIA_LEAD_MASK),
            },
        )

        # 读取 ECG 波形和注释（annotations）
        x_in_mV, fs_in_wave = read_ecg_mV(dataset_root, rec)  # x_in_mV：原始波形（mV），fs_in_wave：读到的采样率
        # 如果 rdrecord/波形读取到的采样率与 header 不一致，给警告（可能是 header/读取逻辑差异）
        if abs(fs_in_wave - fs_in) > 1e-6 and args.verbose:
            print(f"[warn] rdrecord fs mismatch for {rec.record}: header={fs_in} wave={fs_in_wave}")

        ann = read_annotations(dataset_root, rec)  # ann.sample / ann.symbol / ann.aux_note
        # 把注释的 sample（输入域索引）映射到输出域索引（对应 fs_out 的时间轴）
        ann_samples_out = resample_samples(ann.sample, fs_in=fs_in, fs_out=fs_out)

        # 对 ECG 波形做真正的重采样（输出域采样率 = fs_out）
        x_out_mV = resample_signal_polyphase(x_in_mV, fs_in=fs_in, fs_out=fs_out)
        # 适配到标准 12 导联格式（Icentia11k：仅 Lead I 有效，其余补零）
        signal_12, _lead_mask, _source_leads = adapt_icentia11k_to_12lead(x_out_mV)

        # 为当前 segment 准备 marker 列表（全部转到“输出域的 global_sample 时间轴”）
        markers: List[Dict] = []  # 每个元素是一个 marker 字典，稍后会按时间排序并逐个 push
        # zip：把每条注释的 (sample_out, symbol, aux_note) 组合在一起遍历
        for s_out, symbol, aux in zip(ann_samples_out.tolist(), ann.symbol, ann.aux_note):
            aux = safe_aux_note(aux)  # 清理 aux_note（确保是可用字符串）
            marker: Dict = {  # 每条注释都变成一个 kind=annotation 的事件
                "kind": "annotation",  # marker 类型：注释事件
                "global_sample": int(global_sample_offset + int(s_out)),  # 注释发生的全局采样点（输出域）
                "record": rec.record,  # 来自哪个 record
                "segment_id": int(rec.segment_id),  # 来自哪个 segment
                "sample_out_record": int(s_out),  # 在当前 record 内的输出域采样点索引（便于局部对齐）
                "symbol": symbol,  # WFDB 的符号（例如搏动符号、'+' 等）
            }
            # aux_note 通常携带节律信息，例如 "(AFIB" 之类（格式取决于数据集）
            if aux:
                marker["aux_note"] = aux  # 原始 aux_note 也保留，方便调试
                # 这里的判断：如果 aux 以 "(" 开头但不以 ")" 结尾，说明它可能是一个节律标签片段
                # （这取决于 Icentia11k 的标注习惯/工具输出）
                if aux.startswith("(") and not aux.endswith(")"):
                    marker["rhythm_label_raw"] = aux[1:]  # 去掉左括号，得到原始节律标签
                    marker["rhythm_label_norm"] = normalize_rhythm_label(aux[1:])  # 做一次归一化，便于下游统一处理
            # symbol="+" 在 WFDB 注释里常用于“节律/辅助注释”（常配合 aux_note），不是一个具体的搏动（beat），所以不做 beat 归一化
            if symbol != "+":
                marker["beat_symbol_norm"] = normalize_beat_symbol(symbol)  # 规范化搏动符号（便于下游统计/分类）
            markers.append(marker)  # 把 marker 放入列表，稍后统一排序发送

        markers.sort(key=lambda m: int(m["global_sample"]))  # 按发生时间排序，确保发送顺序正确
        marker_idx = 0  # 当前已经发送到第几个 marker（在 while chunk 循环中推进）

        # 按 chunk 把波形推送出去（模拟实时流）
        n = int(signal_12.shape[0])  # 当前 segment 的输出域总样本数
        i = 0  # 当前已发送到输出波形的第 i 个样本（在本 segment 内）
        chunk_count = 0  # 已发送的 chunk 数（用于 watermark_every）
        while i < n:
            # 如果设置了 --max-seconds，并且已经达到上限，则停止 streaming
            if max_samples and streamed_samples >= max_samples:
                break

            chunk = signal_12[i : i + chunk_samples, :]  # 取出一个 chunk（二维：samples x leads）
            if chunk.size == 0:  # 防御性判断：避免空 chunk
                break
            # pylsl 的 push_chunk 期望形状为 (n_samples, n_channels)
            outlet_ecg.push_chunk(chunk)
            '''
            • ECG 是在 push_chunk 那一行真正发出去的，不走 _push_marker()。

            关键位置在这里：

            - 创建 ECG 流（float32 连续波形流）：ECG_Model/code_for_stream/sender_icentia_lsl.py:125 到 ECG_Model/code_for_stream/sender_icentia_lsl.py:132（StreamInfo(... channel_format="float32", nominal_srate=fs_out ...)）
            - 创建 ECG 的 outlet：ECG_Model/code_for_stream/sender_icentia_lsl.py:159（outlet_ecg = pylsl.StreamOutlet(info_ecg, ...)）
            - 发送 ECG 波形数据（真正“推波形”就在这一句）：ECG_Model/code_for_stream/sender_icentia_lsl.py:349
                - chunk = x_out_mV[i : i + chunk_samples]（这是从重采样后的 ECG 波形里切出来的一段）
                - outlet_ecg.push_chunk(chunk.reshape(-1, 1))（把这一段作为 1 通道的 chunk 发到 LSL）

            你之所以“只看到 marker 逻辑”，是因为：

            - marker 发送被封装成 _push_marker(outlet_ann, ...)（ECG_Model/code_for_stream/sender_icentia_lsl.py:164 起），到处都在调用；
            - ECG 发送没有封装函数，直接在 while 循环里一行 outlet_ecg.push_chunk(...)，很容易扫过去没注意到。
            '''

            chunk_start_global = global_sample_offset + i  # 当前 chunk 的全局起点（输出域）
            chunk_end_global = chunk_start_global + int(chunk.shape[0])  # 当前 chunk 的全局终点（输出域，开区间）
            # 把发生在这个 chunk 覆盖范围内的 marker 依次发出去（marker 的 global_sample < chunk_end_global）
            while marker_idx < len(markers) and int(markers[marker_idx]["global_sample"]) < chunk_end_global:
                _push_marker(outlet_ann, markers[marker_idx])  # 发送一个注释 marker
                marker_idx += 1  # 指向下一个 marker
            chunk_count += 1  # 已发送 chunk 数 +1
            # 按设置的频率发送 watermark：kind=ecg_chunk_end
            if chunk_count % watermark_every == 0:
                _push_marker(
                    outlet_ann,
                    {
                        "kind": "ecg_chunk_end",  # marker 类型：ECG chunk 结束（水位线）
                        "global_sample": int(chunk_end_global),  # 水位线位置 = 当前 chunk 末尾的 global_sample
                    },
                )

            # 控制发送节奏（仿实时播放）
            if float(args.speed) > 0:
                # 发送 chunk 后 sleep：chunk 的时长 = chunk_samples / fs_out，再除以 speed 得到加速后的 sleep
                time.sleep(float(chunk.shape[0]) / (fs_out * float(args.speed)))

            i += int(chunk.shape[0])  # segment 内偏移推进
            streamed_samples += int(chunk.shape[0])  # 全局累计发送样本数推进

        # 如果因为 --max-seconds 提前停止，只累加“实际发送的样本数”（i）到全局 offset
        streamed_in_segment = i  # 本 segment 实际发送了多少输出样本
        global_sample_offset += streamed_in_segment  # 更新全局 offset（下一 segment 的 global_sample 从这里接着算）

        # 无论 watermark_every > 1 还是中途截断，都在 segment 结束时再发一次最终水位线
        # 这样接收端可以确定“当前已经完整收到到 global_sample_offset 了”
        _push_marker(
            outlet_ann,
            {
                "kind": "ecg_chunk_end",  # 最终水位线
                "global_sample": int(global_sample_offset),  # 位置 = segment 实际发送末尾（全局）
            },
        )

        _push_marker(
            outlet_ann,
            {
                "kind": "segment_end",  # marker 类型：segment 结束
                "global_sample": int(global_sample_offset),  # segment 结束时的全局采样点
                "record": rec.record,  # 哪个 record 结束
                "segment_id": int(rec.segment_id),  # 哪个 segment id 结束
                "sig_len_out": int(n),  # 输出域该 segment 的理论长度（重采样后的总长度）
                "streamed_out": int(streamed_in_segment),  # 实际发送出去的长度（可能被 --max-seconds 截断）
                "truncated": bool(streamed_in_segment < n),  # 是否截断（实际 < 理论）
            },
        )

        # 如果达到 --max-seconds 上限，跳出最外层 for 循环（不再继续下一个 segment）
        if max_samples and streamed_samples >= max_samples:
            if args.verbose:
                print("[stop] reached --max-seconds")
            break

    _push_marker(
        outlet_ann,
        {
            "kind": "session_end",  # marker 类型：会话结束
            "global_sample": int(global_sample_offset),  # 会话结束时的全局采样点
            "elapsed_sec_wall": float(time.time() - t_start),  # 墙钟耗时（实际运行花了多少秒）
            "speed": float(args.speed),  # 本次会话的播放速度配置（便于记录）
        },
    )
    if args.verbose:
        print(f"[done] streamed_samples={streamed_samples} global_end={global_sample_offset}")  # 打印总结信息
    return 0  # main 返回 0 表示正常结束


if __name__ == "__main__":
    raise SystemExit(main())  # 以脚本方式运行时执行 main，并把返回码作为进程退出码
