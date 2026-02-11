import torch.nn as nn

class ECGReprogrammer(nn.Module):
    def __init__(self, ecg_dim, llm_dim):
        super().__init__()
        self.proj = nn.Linear(ecg_dim, llm_dim)

    def forward(self, x):
        return self.proj(x)
