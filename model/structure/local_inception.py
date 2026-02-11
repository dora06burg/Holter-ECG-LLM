import torch
import torch.nn as nn
from tsai.models.InceptionTime import InceptionTime

class InceptionLocalEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # c_in=1 表示单导联心电输入
        self.net = InceptionTime(c_in=1, c_out=emb_dim)

    def forward(self, beats):
        """
        beats: 形状为 (N_beats, beat_len) 的 Tensor，由 Splitter 直接传入
        """
        # 处理异常空输入（例如当前窗口全是基线直线，没有心跳）
        if beats.numel() == 0:
            return torch.empty((0, self.net.c_out)).to(beats.device)
            
        # InceptionTime 需要的输入形状是: (Batch_size, Channels, Sequence_Length)
        # 所以我们需要在第 1 维（Channels）增加一个维度
        # x 形状变为: (N_beats, 1, beat_len)
        x = beats.unsqueeze(1)
        
        # O(1) 复杂度的前向传播：让 GPU 并行处理这 N 个心搏
        # feats 形状: (N_beats, emb_dim)
        feats = self.net(x)
        
        return feats