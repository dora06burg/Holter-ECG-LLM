import torch
from tqdm import tqdm
from data.stream import ECGStream
from structure.splitter import MultiScaleSplitter

class StreamingTrainer:
    def __init__(self, model, scorer, optimizer, fs, writer):
        self.model = model
        self.scorer = scorer
        self.optimizer = optimizer
        self.splitter = MultiScaleSplitter(fs)
        self.writer = writer
        self.step = 0

    def train_record(self, ecg):
        stream = ECGStream(ecg, fs=128)

        for window in stream:
            beats = self.splitter.split(window)
            if len(beats) < 3:
                continue

            importance = torch.tensor([
                self.scorer.score(b) for b in beats
            ])

            emb = self.model(beats, importance)
            loss = emb.norm()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("loss/stream", loss.item(), self.step)
            self.step += 1
