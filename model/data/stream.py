import numpy as np

class ECGStream:
    def __init__(self, signal, fs, window_sec=30, stride_sec=5):
        self.signal = signal
        self.fs = fs
        self.W = int(window_sec * fs)
        self.S = int(stride_sec * fs)

    def __iter__(self):
        for start in range(0, len(self.signal) - self.W, self.S):
            yield self.signal[start:start + self.W]
