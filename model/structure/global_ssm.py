import torch
import torch.nn as nn

class GlobalSSMEncoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        depth=3,
        ssm_type="s4",   # "s4" or "mamba"
        **kwargs
    ):
        super().__init__()
        self.ssm_type = ssm_type.lower()

        if self.ssm_type == "s4":
            from s4.models.s4.s4 import S4Block
            self.layers = nn.ModuleList([
                S4Block(
                    d_model=emb_dim,
                    transposed=False,
                    **kwargs
                )
                for _ in range(depth)
            ])

        elif self.ssm_type == "mamba":
            from mamba_ssm import Mamba
            self.layers = nn.ModuleList([
                Mamba(
                    d_model=emb_dim,
                    **kwargs
                )
                for _ in range(depth)
            ])

        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")

        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, importance):
        """
        x: [B, T, C] or [T, C]
        importance: [B, T] or [T]
        """

        # ---------- 1. 统一成 batch 形式 ----------
        if x.dim() == 2:
            x = x.unsqueeze(0)              # [1, T, C]
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)  # [1, T]

        B, T, C = x.shape

        # ---------- 2. importance gating ----------
        gate = importance / (
            importance.max(dim=1, keepdim=True)[0] + 1e-6
        )                                   # [B, T]

        x = x * gate.unsqueeze(-1)          # [B, T, C]

        # ---------- 3. SSM layers ----------
        if self.ssm_type == "s4":
            for layer in self.layers:
                x, _ = layer(x)

        elif self.ssm_type == "mamba":
            for layer in self.layers:
                x = layer(x)

        # ---------- 4. normalization ----------
        x = self.norm(x)

        return x.squeeze(0) if B == 1 else x
