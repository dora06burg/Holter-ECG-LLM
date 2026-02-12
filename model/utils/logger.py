from torch.utils.tensorboard import SummaryWriter

def get_logger(logdir):
    """
    初始化 Tensorboard 日志记录器。
    
    Args:
        logdir (str): 日志文件保存的路径 (例如 "runs/stream_ecg")
        
    Returns:
        SummaryWriter: 用于写入 Loss 曲线和指标的对象
    """
    return SummaryWriter(logdir)