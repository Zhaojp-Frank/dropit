import torch
from .dropit import to_dropit, DropITer

class Contiguous(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor):
        return x.contiguous()