import torch
import neurokit2 as nk
import numpy as np

class MultiScaleSplitter:
    def __init__(self, fs):
        self.fs = fs
        # 预先计算好 R 波前后的点数，保证所有 Beat 长度绝对一致
        self.pre_r = int(0.3 * self.fs)
        self.post_r = int(0.4 * self.fs)
        self.beat_len = self.pre_r + self.post_r

    def split_beats(self, window):
        """
        全矩阵并行化提取心搏
        window: 1D numpy array 或者是形状为 (L,) 的 Tensor
        """
        if isinstance(window, torch.Tensor):
            window = window.cpu().numpy()
            
        # 1. 抓取 R 波峰值索引
        _, rpeaks = nk.ecg_peaks(window, sampling_rate=self.fs)
        peaks = rpeaks["ECG_R_Peaks"]
        
        # 2. 过滤掉靠近边缘的“残缺心搏”
        valid_peaks = peaks[(peaks >= self.pre_r) & (peaks + self.post_r <= len(window))]
        
        if len(valid_peaks) == 0:
            # 极端情况处理：当前窗口没有有效心跳（例如严重停搏或噪声）
            return torch.empty((0, self.beat_len))
            
        # 3. 核心魔法：利用广播机制生成二维索引矩阵
        # base_idx shape: (beat_len,)
        base_idx = np.arange(-self.pre_r, self.post_r)
        
        # idx_matrix shape: (N_beats, beat_len)
        # 这一步瞬间算出了所有心搏在原序列中的位置
        idx_matrix = valid_peaks[:, None] + base_idx
        
        # 4. 一次性提取所有心搏
        # beats_np shape: (N_beats, beat_len)
        beats_np = window[idx_matrix]
        
        # 返回标准的 PyTorch Tensor
        return torch.tensor(beats_np, dtype=torch.float32)

    def split(self, window):
        return self.split_beats(window)