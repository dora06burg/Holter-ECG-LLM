import numpy as np

def normalize(ecg):
    ecg = ecg - np.mean(ecg)
    ecg = ecg / (np.std(ecg) + 1e-8) # 1e-8 防除零报错很严谨
    return ecg