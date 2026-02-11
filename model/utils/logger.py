from torch.utils.tensorboard import SummaryWriter

def get_logger(logdir):
    return SummaryWriter(logdir)
