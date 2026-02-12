import torch
import torch.nn as nn

class GlobalSSMEncoder(nn.Module):
    """
    全局节律编码器 (Global Rhythm Encoder)。
    对应架构图中的【全局节律 Encoder】。
    
    功能：
    1. 接收局部形态特征 (Local Features)。
    2. 利用重要性得分 (Importance) 进行 Gating 加权。
    3. 使用 Mamba/S4 进行长程时序建模，捕捉心律失常的上下文规律。
    """
    def __init__(
        self,
        emb_dim,
        depth=3,
        ssm_type="mamba",
        **kwargs
    ):
        super().__init__()
        self.ssm_type = ssm_type.lower()

        # 根据配置加载 Mamba 或 S4 层
        if self.ssm_type == "s4":
            from s4.models.s4.s4 import S4Block
            self.layers = nn.ModuleList([
                S4Block(d_model=emb_dim, transposed=False, **kwargs)
                for _ in range(depth)
            ])
        elif self.ssm_type == "mamba":
            from mamba_ssm import Mamba
            self.layers = nn.ModuleList([
                Mamba(d_model=emb_dim, **kwargs)
                for _ in range(depth)
            ])
        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")

        self.norm = nn.LayerNorm(emb_dim)

    # 【修复 1】：补齐状态初始化方法
    def init_stream_state(self, max_seqlen, batch_size):
        """
        初始化流式推理所需的记忆体 (Cache)。
        对于 Mamba，这是 InferenceParams；对于 S4，这是 None (S4 需手动管理 List)。
        """
        if self.ssm_type == "mamba":
            from mamba_ssm.utils.generation import InferenceParams
            return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
        elif self.ssm_type == "s4":
            return None 

    # 【修复 2】：统一使用 state 参数名
    def forward(self, x, importance, state=None):
        """
        Args:
            x: [Batch, Time, Channel] 心跳的局部特征序列
            importance: [Batch, Time] 每个心跳的异常得分
            state: 上一时刻传递过来的流式记忆体
        """
        # ---------- 1. 统一成 batch 形式 ----------
        # 确保输入是 3D 张量 [B, T, C]
        if x.dim() == 2:
            x = x.unsqueeze(0)              # [T, C] -> [1, T, C]
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)  # [T] -> [1, T]

        B, T, C = x.shape

        # ---------- 2. Importance Gating (软注意力筛选) ----------
        # 对应架构图中那条虚线箭头：利用异常分控制特征流
        # 归一化得分到 0~1 之间
        gate = importance / (importance.max(dim=1, keepdim=True)[0] + 1e-6)
        # 加权：异常心跳特征被放大，正常心跳特征被抑制
        x = x * gate.unsqueeze(-1)          # [B, T, C]

        # ---------- 3. SSM layers (长程记忆建模) ----------
        if self.ssm_type == "s4":
            new_states = []
            for i, layer in enumerate(self.layers):
                # 逐层取出对应的状态
                layer_state = state[i] if state is not None else None
                x, s = layer(x, state=layer_state) 
                new_states.append(s)
            state = new_states # 更新状态列表
            
        elif self.ssm_type == "mamba":
            # Mamba 的 InferenceParams 会在内部自动处理层级索引
            for layer in self.layers:
                x = layer(x, inference_params=state)
            
            # 手动推进 Mamba 的时间指针 (Sequence Length Offset)
            if state is not None:
                state.seqlen_offset += x.shape[1]

        x = self.norm(x)
        
        # 【修复核心】：删掉 squeeze 操作！
        # 无论 B 是几，永远返回 [B, T, C] 格式，保证 trainer 里的 dim=1 永远指向 Time
        return x, state