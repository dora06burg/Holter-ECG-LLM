import numpy as np

def normalize(ecg):
    """
    Z-score 标准化 (Zero-mean Unit-variance Normalization).
    
    Args:
        ecg: numpy array (一段心电信号)
        
    Returns:
        归一化后的信号，均值为 0，标准差为 1。
    """
    # 减去均值，消除直流偏置 (DC Offset)
    ecg = ecg - np.mean(ecg)
    
    # 除以标准差，统一幅度范围
    # + 1e-8 是为了防止标准差为 0 (例如死人直线心电图) 导致除以零报错
    ecg = ecg / (np.std(ecg) + 1e-8) 
    
    return ecg