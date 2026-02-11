import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.LSTM(1, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, 1, batch_first=True)

    def forward(self, x):
        z, _ = self.enc(x)
        out, _ = self.dec(z)
        return out

class AbnormalityScorer:
    def __init__(self, ae):
        self.ae = ae.eval()

    def score(self, beat):
        x = torch.tensor(beat).float().unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            recon = self.ae(x)
        return torch.mean((x - recon) ** 2).item()
