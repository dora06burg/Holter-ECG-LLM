import torch
import neurokit2 as nk
import numpy as np

class MultiScaleSplitter:
    """
    多尺度切分器 (Multi-scale Splitter)。
    对应架构图中的【多尺度时序切分模块 -> 心搏级 Beat-level】。
    
    作用：
    将连续的 5 分钟心电图窗口，利用 R 波定位，
    极速切分为数百个独立的心搏 (Beats)。
    """
    def __init__(self, fs):
        self.fs = fs
        # 预先计算好 R 波前后的点数，保证所有 Beat 长度绝对一致
        # R波前 0.3秒，后 0.4秒，这是医学上观察 P-QRS-T 的常用窗口
        self.pre_r = int(0.3 * self.fs)
        self.post_r = int(0.4 * self.fs)
        self.beat_len = self.pre_r + self.post_r

    def split_beats(self, window):
        """
        全矩阵并行化提取心搏 (Zero For-Loop Implementation)。
        
        Args:
            window: 1D numpy array 或者是形状为 (L,) 的 Tensor (当前 5 分钟数据)
            
        Returns:
            beats: (N_beats, beat_len) 的 Tensor
        """
        if isinstance(window, torch.Tensor):
            window = window.cpu().numpy()
            
        # 1. 使用 neurokit2 抓取 R 波峰值索引 (CPU操作)
        _, rpeaks = nk.ecg_peaks(window, sampling_rate=self.fs)
        peaks = rpeaks["ECG_R_Peaks"]
        
        # 2. 边界检查：过滤掉靠近窗口边缘、无法截取完整长度的“残缺心搏”
        valid_peaks = peaks[(peaks >= self.pre_r) & (peaks + self.post_r <= len(window))]
        
        if len(valid_peaks) == 0:
            # 极端情况处理：当前窗口没有有效心跳（例如严重停搏或噪声）
            return torch.empty((0, self.beat_len))
            
        # 3. 核心魔法：利用 NumPy 广播机制生成二维索引矩阵
        # base_idx: [-30, -29, ..., 0, ..., 40] (相对索引)
        base_idx = np.arange(-self.pre_r, self.post_r)
        
        # idx_matrix: (N_beats, 1) + (beat_len,) -> (N_beats, beat_len)
        # 这一步瞬间算出了所有心搏在原序列中的绝对索引位置，避免了 Python for 循环
        idx_matrix = valid_peaks[:, None] + base_idx
        
        # 4. 一次性提取所有心搏 (Fancy Indexing)
        # beats_np shape: (N_beats, beat_len)
        beats_np = window[idx_matrix]
        
        # 返回标准的 PyTorch Tensor
        return torch.tensor(beats_np, dtype=torch.float32)

    def split(self, window):
        """对外暴露的统一接口"""
        return self.split_beats(window)