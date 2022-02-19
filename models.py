import torch
from torch import nn
from torchmetrics.functional import accuracy
from torch.functional import F
from pytorch_lightning import LightningModule
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from layers import to_dropit, DropITer, Contiguous
import torchvision.models
import timm

class Classifier(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        arch = cfg.MODEL.ARCH
        if "resnet" in arch:
            self.model = getattr(torchvision.models, arch)(num_classes=cfg.MODEL.NUM_CLASSES)
            if "cifar100" in cfg.DATASET:
                self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                self.model.maxpool = nn.Identity()
        elif "vit" in arch:
            self.model = timm.create_model(arch, pretrained=True, num_classes=cfg.MODEL.NUM_CLASSES)
            self.model.pre_logits = Contiguous()
        if cfg.MODEL.DROPIT:
            D = cfg.DROPIT.D if hasattr(cfg.DROPIT, "D") else None
            self.dropiter = DropITer(cfg.DROPIT.STRATEGY, cfg.DROPIT.GAMMA, D)
            to_dropit(self.model, self.dropiter)
        
        self.cfg = cfg
        self.arch = arch
    
    def training_step(self, batch, idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        dataset = self.trainer.datamodule.__class__.__name__
        self.log(f"{dataset} loss", loss, rank_zero_only=True)
        return loss
    
    def configure_optimizers(self):
        cfg = self.cfg
        model = self.model
        if "resnet18" in self.arch:
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR,
                      momentum=0.9, weight_decay=cfg.SOLVER.WEIGHT_DECAY) 
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                milestones=cfg.SOLVER.MILESTONES, gamma=cfg.SOLVER.GAMMA) 
        elif "vit" in self.arch:
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR,
                      momentum=0.9, weight_decay=0)
            lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS, 
                max_epochs=cfg.SOLVER.MAX_EPOCHS, warmup_start_lr=cfg.SOLVER.LR*0.1)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, idx):
        x, y = batch
        return self.model(x).cpu(), y.cpu()
    
    def validation_epoch_end(self, outputs) -> None:
        scores, labels = list(zip(*outputs))
        scores, labels = torch.cat(scores), torch.cat(labels)
        if self.trainer.num_gpus > 1:
            scores, labels = self.all_gather([scores, labels])
            scores = scores.view(-1, scores.shape[-1])
            labels = labels.view(-1)
        acc1 = accuracy(scores, labels, top_k=1)
        acc5 = accuracy(scores, labels, top_k=5)
        dataset = self.trainer.datamodule.__class__.__name__
        self.log(f"{dataset} top1 accuracy", acc1, rank_zero_only=True)
        self.log(f"{dataset} top5 accuracy", acc5, rank_zero_only=True)
        return dict(acc1=acc1, acc5=acc5)

def build_model(cfg):
    return Classifier(cfg)