import torch
import torch.nn as nn

class GlobalSSMEncoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        depth=3,
        ssm_type="mamba",
        **kwargs
    ):
        super().__init__()
        self.ssm_type = ssm_type.lower()

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
        if self.ssm_type == "mamba":
            from mamba_ssm.utils.generation import InferenceParams
            return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
        elif self.ssm_type == "s4":
            return None 

    # 【修复 2】：统一使用 state 参数名
    def forward(self, x, importance, state=None):
        # ---------- 1. 统一成 batch 形式 ----------
        if x.dim() == 2:
            x = x.unsqueeze(0)              # [1, T, C]
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)  # [1, T]

        B, T, C = x.shape

        # ---------- 2. importance gating ----------
        gate = importance / (importance.max(dim=1, keepdim=True)[0] + 1e-6)
        x = x * gate.unsqueeze(-1)          # [B, T, C]

        # ---------- 3. SSM layers ----------
        if self.ssm_type == "s4":
            new_states = []
            for i, layer in enumerate(self.layers):
                layer_state = state[i] if state is not None else None
                x, s = layer(x, state=layer_state) 
                new_states.append(s)
            state = new_states 
            
        elif self.ssm_type == "mamba":
            for layer in self.layers:
                x = layer(x, inference_params=state)
            
            if state is not None:
                state.seqlen_offset += x.shape[1]

        x = self.norm(x)
        
        # 【修复核心】：删掉 squeeze 操作！
        # 无论 B 是几，永远返回 [B, T, C] 格式，保证 trainer 里的 dim=1 永远指向 Time
        return x, state