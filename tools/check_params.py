import torch
from pytorch_lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter
import os

def load_state_dict(model):
    return torch.load(os.path.join(model), map_location="cpu")["state_dict"]

def dist(a, b):
    return torch.dist(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()

if __name__ == "__main__":
    writer = SummaryWriter()
    seed_everything(42, workers=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    baseline = load_state_dict("outputs/kinetics400/x3d_s/lightning_logs/version_6/checkpoints/epoch=299-step=278399.ckpt")
    pg4t = load_state_dict("outputs/kinetics400/temporal_conv2d_epoch=299-step=278399.ckpt")
    pg4c = load_state_dict("outputs/kinetics400/x3d_s_pg4c_conv1-5+wd_stridex2/lightning_logs/version_0/checkpoints/epoch=299-step=278399.ckpt")
    pg4s = load_state_dict("outputs/kinetics400/spatail_conv_epoch=299-step=278399.ckpt")
    for b, t, c, s in zip(baseline, pg4t, pg4c, pg4s):
        writer.add_histogram(b, baseline[b], 0)
        writer.add_histogram(b, pg4c[c], 1)
        writer.add_histogram(b, pg4t[t], 2)
        writer.add_histogram(b, pg4s[s], 3)
    writer.close()