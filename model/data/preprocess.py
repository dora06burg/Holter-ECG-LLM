import numpy as np

def normalize(ecg):
    ecg = ecg - np.mean(ecg)
    ecg = ecg / (np.std(ecg) + 1e-8)
    return ecg
