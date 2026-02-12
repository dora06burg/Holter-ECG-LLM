import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InfoNCELoss(nn.Module):
    def __init__(self, initial_temp=0.07):
        super().__init__()
        # 【核心细节】：可学习的温度参数 (Learnable Temperature)
        # 顶会 Reviewer 看到这个细节会知道你们是内行，它能动态调节拉伸/推开的力度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / initial_temp))

    def forward(self, ecg_features, text_features):
        """
        ecg_features: [Batch, LLM_Dim] (当前窗口的心电全局特征向量)
        text_features: [Batch, LLM_Dim] (医生诊断文本的 Embedding)
        """
        # 1. L2 归一化，把所有向量投影到同一个超球面上
        ecg_features = F.normalize(ecg_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 2. 限制温度参数防止溢出
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        # 3. 计算余弦相似度矩阵 (Cosine Similarity Matrix)
        # logits 矩阵对角线上是正确的匹配，非对角线全是错误的配对
        logits_per_ecg = logit_scale * ecg_features @ text_features.T
        logits_per_text = logits_per_ecg.T

        # 4. 生成正确的标签 (0, 1, 2... Batch_size-1)
        batch_size = ecg_features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=ecg_features.device)

        # 5. 计算对称交叉熵 (Symmetric Cross Entropy)
        loss_ecg = F.cross_entropy(logits_per_ecg, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        
        # 最终的对比损失
        return (loss_ecg + loss_text) / 2