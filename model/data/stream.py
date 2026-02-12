import numpy as np
from .preprocess import normalize # 把你的预处理函数导进来

class ECGStream:
    def __init__(self, signal, fs, window_sec=300):
        """
        signal: 1D numpy array
        window_sec: 推荐 300 秒 (5分钟)，充分榨干 GPU 算力
        """
        self.signal = signal
        self.fs = fs
        self.W = int(window_sec * fs)
        # 【修复 1】：强行取消 stride 参数，强制步长 = 窗口大小
        # 保证 Mamba 接收到的时间线是严丝合缝、连续向前的
        self.S = self.W 

    def __len__(self):
        # 【修复 2】：提供准确的 chunk 数量，让 tqdm 进度条完美显示
        # 计算可以切出多少个完整的 window
        return len(range(0, len(self.signal), self.S))

    def __iter__(self):
        for start in range(0, len(self.signal), self.S):
            end = start + self.W
            chunk = self.signal[start:end]
            
            # 丢弃最后不足 3 秒的碎片段，防止特征提取报错
            if len(chunk) < self.fs * 3:
                continue
                
            # 【修复 3】：实时动态归一化 (Local Normalization)
            # 消除 Holter 长程信号中极其常见的基线漂移 (Baseline Wander)
            chunk = normalize(chunk)
            
            yield chunk