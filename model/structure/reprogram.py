# 替换掉你原来的 reprogram.py
import torch
import torch.nn as nn

class ECGReprogrammer(nn.Module):
    def __init__(self, ecg_dim, llm_dim, hidden_dim=None):
        super().__init__()
        # 默认隐藏层维度扩大一倍，给予模型更充足的映射空间
        if hidden_dim is None:
            hidden_dim = ecg_dim * 2

        # 摒弃单层 Linear，使用经典的 Projector 结构
        self.proj = nn.Sequential(
            nn.Linear(ecg_dim, hidden_dim),
            nn.GELU(),                  # 引入非线性，这对于跨模态极其重要
            nn.LayerNorm(hidden_dim),   # 稳定数值分布，防止在输入给 LLM 时造成梯度爆炸
            nn.Linear(hidden_dim, llm_dim)
        )

    def forward(self, x):
        """
        x: [B, Seq_len, ecg_dim] (Mamba 吐出来的心电特征)
        返回: [B, Seq_len, llm_dim] (可以直接喂给 LLM 的虚拟 Token)
        """
        return self.proj(x)