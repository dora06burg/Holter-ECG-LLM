import torch.nn as nn
from .local_inception import InceptionLocalEncoder
from .global_ssm import GlobalSSMEncoder

class HierarchicalECGEncoder(nn.Module):
    """
    层次化 ECG 编码器 (Hierarchical ECG Encoder)。
    对应架构图中中间的大方框。
    
    职责：
    组合【局部形态 Encoder】和【全局节律 Encoder】，
    对外提供统一的接口。
    """
    def __init__(
        self,
        emb_dim=128,
        ssm_type="s4", # 建议默认改为 mamba 保持一致
        ssm_depth=3,
        **ssm_kwargs
    ):
        super().__init__()
        # 局部特征提取器 (InceptionTime)
        self.local = InceptionLocalEncoder(emb_dim)
        # 全局时序建模器 (Mamba/S4)
        self.global_enc = GlobalSSMEncoder(
            emb_dim=emb_dim,
            depth=ssm_depth,
            ssm_type=ssm_type,
            **ssm_kwargs
        )

    # 【修复 1】：添加状态初始化代理方法
    def init_stream_state(self, max_seqlen, batch_size):
        """代理调用底层的初始化方法"""
        return self.global_enc.init_stream_state(max_seqlen, batch_size)

    # 【修复 2】：接收 state 并传递给底层
    def forward(self, beats, importance, state=None):
        """
        前向传播流程：
        1. Beats -> Local Encoder -> Local Features (微观形态)
        2. Local Features + Importance -> Global Encoder -> Global Embeddings (宏观节律)
        
        Args:
            beats: [B, T, beat_len] 切分好的心跳波形
            importance: [B, T] 异常得分
            state: 流式记忆体
        """
        # Step 1: 提取局部特征
        local_feat = self.local(beats)
        
        # Step 2: 提取全局节律特征 (包含 Gating 和 记忆更新)
        # 返回更新后的特征和状态
        return self.global_enc(local_feat, importance, state=state)