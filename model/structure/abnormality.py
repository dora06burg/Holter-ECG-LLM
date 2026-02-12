import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    """
    基于 LSTM 的自编码器，用于学习“正常心跳”的形态。
    原理：仅在正常心跳上预训练，推理时如果遇到异常心跳，重构误差(MSE)会很大。
    """
    def __init__(self, hidden=64):
        super().__init__()
        # 编码器：将时序点压缩为 hidden 向量
        self.enc = nn.LSTM(1, hidden, batch_first=True)
        # 解码器：将 hidden 向量还原回原始时序点
        self.dec = nn.LSTM(hidden, 1, batch_first=True)

    def forward(self, x):
        # x shape: [Batch, SeqLen, 1]
        z, _ = self.enc(x)
        out, _ = self.dec(z)
        return out

class AbnormalityScorer:
    """
    异常打分器：计算输入心跳的重构误差 (MSE)。
    误差越大 -> 说明该心跳越不像正常窦性心律 -> 重要性(Importance)越高。
    """
    def __init__(self, ae):
        self.ae = ae.eval() # 始终处于评估模式，冻结 Dropout/BN

    def score(self, beat):
        """单次处理一个心跳 (主要用于测试)"""
        x = torch.tensor(beat).float().unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            recon = self.ae(x)
        return torch.mean((x - recon) ** 2).item()

    def score_batch(self, beats_tensor):
        """
        全矩阵并行计算一批心跳的异常得分 (用于训练时的 Gating)。
        
        Args:
            beats_tensor: (N_beats, beat_len) 直接在 GPU 上的二维张量
            
        Returns:
            mse: (N_beats,) 每个心跳的均方误差得分
        """
        # 增加特征维度，适应 LSTM 输入要求
        # Shape: (N_beats, beat_len) -> (N_beats, beat_len, 1)
        x = beats_tensor.unsqueeze(-1)
        
        with torch.no_grad():
            recon = self.ae(x)
            # 计算每个心搏的 MSE 误差
            # dim=[1, 2] 表示在 时间步 和 特征 维度上求平均，保留 Batch 维度
            mse = torch.mean((x - recon) ** 2, dim=[1, 2])
        return mse