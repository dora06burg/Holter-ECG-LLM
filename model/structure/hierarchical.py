import torch.nn as nn
from .local_inception import InceptionLocalEncoder
from .global_ssm import GlobalSSMEncoder

class HierarchicalECGEncoder(nn.Module):
    def __init__(
        self,
        emb_dim=128,
        ssm_type="s4",
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

    def forward(self, beats, importance):
        """
        beats: [B, T, beat_len] or already embedded
        importance: [B, T]
        """
        local_feat = self.local(beats)
        return self.global_enc(local_feat, importance)
