import torch.nn as nn
from .local_inception import InceptionLocalEncoder
from .global_ssm import GlobalSSMEncoder

class HierarchicalECGEncoder(nn.Module):
    def __init__(
        self,
        emb_dim=128,
        ssm_type="s4", # 建议默认改为 mamba 保持一致
        ssm_depth=3,
        **ssm_kwargs
    ):
        super().__init__()
        self.local = InceptionLocalEncoder(emb_dim)
        self.global_enc = GlobalSSMEncoder(
            emb_dim=emb_dim,
            depth=ssm_depth,
            ssm_type=ssm_type,
            **ssm_kwargs
        )

    # 【修复 1】：添加状态初始化代理方法
    def init_stream_state(self, max_seqlen, batch_size):
        return self.global_enc.init_stream_state(max_seqlen, batch_size)

    # 【修复 2】：接收 state 并传递给底层
    def forward(self, beats, importance, state=None):
        """
        beats: [B, T, beat_len]
        importance: [B, T]
        state: 流式记忆体
        """
        local_feat = self.local(beats)
        # 返回更新后的特征和状态
        return self.global_enc(local_feat, importance, state=state)