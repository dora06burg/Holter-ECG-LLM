import numpy as np
from .preprocess import normalize # 导入预处理函数

class ECGStream:
    """
    ECG 流式迭代器 (ECG Streaming Iterator)。
    对应架构图的【多尺度时序切分模块 -> 长时趋势区间 Minutes】。
    
    作用：
    将巨大的 24小时数组，按顺序切分为一个个小的 5分钟窗口 (Window)。
    就像视频流播放一样，让模型一段一段地吃数据。
    """
    def __init__(self, signal, fs, window_sec=300):
        """
        Args:
            signal: 1D numpy array (24h 完整数据)
            window_sec: 窗口时长，默认 300秒 (5分钟)。
                        这个长度经过权衡，既能榨干 GPU 并行能力，又不至于爆显存。
        """
        self.signal = signal
        self.fs = fs
        self.W = int(window_sec * fs) # 窗口长度 (采样点数)
        
        # 【修复 1】：强行取消 stride 参数，强制步长(Stride) = 窗口大小(Window)
        # 理由：Mamba/RNN 是具有状态记忆的模型。
        # 如果窗口有重叠 (Overlap)，模型会把重叠部分的时间“重复记忆”，导致时序逻辑错乱。
        # 所以必须是无缝拼接 (Non-overlapping)。
        self.S = self.W 

    def __len__(self):
        # 【修复 2】：提供准确的 Chunk 数量
        # 这让 tqdm 能够显示进度条 (例如: 15% [|||.......])
        return len(range(0, len(self.signal), self.S))

    def __iter__(self):
        # 生成器 (Generator) 逻辑
        for start in range(0, len(self.signal), self.S):
            end = start + self.W
            chunk = self.signal[start:end]
            
            # 丢弃末尾不足 3 秒的碎片段
            # 理由：太短的片段切不出完整心跳，不仅无意义，还可能导致 Splitter 报错。
            if len(chunk) < self.fs * 3:
                continue
                
            # 【修复 3】：局部动态归一化 (Local Normalization)
            # 理由：Holter 记录中，病人的电极接触情况一直在变（睡觉压到、出汗等）。
            # 这导致整个 24h 的基线会剧烈漂移。
            # 如果做全局归一化，局部的小波形会被大漂移掩盖。
            # 做“局部归一化”能强制把当前这 5分钟 的波形拉回到标准范围，让模型看清细节。
            chunk = normalize(chunk)
            
            yield chunk