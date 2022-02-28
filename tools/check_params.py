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
    
    baseline = load_state_dict("outputs/cifar100/resnet18/lightning_logs/version_1/checkpoints/epoch=199-step=77999.ckpt")
    dropit_fast_minik_08 = load_state_dict("outputs/cifar100/resnet18_fastminkx0.8/lightning_logs/version_1/checkpoints/epoch=199-step=77999.ckpt")
    dropit_fast_minik_099 = load_state_dict("outputs/cifar100/resnet18_fastminkx0.99/lightning_logs/version_1/checkpoints/epoch=199-step=77999.ckpt")
    for b, d1, d2 in zip(baseline, dropit_fast_minik_08, dropit_fast_minik_099):
        writer.add_histogram(b, baseline[b], 0)
        writer.add_histogram(b, dropit_fast_minik_08[d1], 1)
        writer.add_histogram(b, dropit_fast_minik_099[d2], 2)
        print(torch.norm((dropit_fast_minik_08[d1]-baseline[b]).float(), 2))
        print(torch.norm((dropit_fast_minik_099[d2]-baseline[b]).float(), 2))
        print("------------")

    writer.close()