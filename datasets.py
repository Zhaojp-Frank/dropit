from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Normalize, ToTensor, 
    RandomCrop, RandomHorizontalFlip, Pad,
    Resize, CenterCrop, RandomResizedCrop
)
from torchvision.datasets import CIFAR100, ImageFolder
from pytorch_lightning import LightningDataModule

class Cifar100DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def train_dataloader(self): 
        cfg = self.cfg
        arch = cfg.MODEL.ARCH
        if "vit" not in arch:
            transform = Compose([
                Pad(4, padding_mode='reflect'),
                RandomHorizontalFlip(),
                RandomCrop(32),
                ToTensor(),
                Normalize((125.3/255, 123/255, 113.9/255), 
                    (63/255, 62.1/255, 66.7/255)),
            ])
        else:
            transform = Compose([
                RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), 
                    (0.5, 0.5, 0.5)),
            ])
        trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
        return DataLoader(trainset, batch_size=cfg.SOLVER.BATCH_SIZE//cfg.NUM_GPUS, shuffle=True, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        cfg = self.cfg
        arch = cfg.MODEL.ARCH
        if "vit" not in arch:
            transform = Compose([
                ToTensor(),
                Normalize((125.3/255, 123/255, 113.9/255), 
                    (63/255, 62.1/255, 66.7/255)),
            ])
        else:
            transform = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), 
                    (0.5, 0.5, 0.5)),
            ])
        testset = CIFAR100(root='./data', train=False, download=True, transform=transform)
        return DataLoader(testset, batch_size=cfg.SOLVER.BATCH_SIZE//cfg.NUM_GPUS, shuffle=False, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

class ImageNetDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        cfg = self.cfg
        dataset = ImageFolder(
            cfg.TRAIN.PATH,
            Compose([
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ])
        )
        return DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE//cfg.NUM_GPUS,
            shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        cfg = self.cfg
        dataset = ImageFolder(
            cfg.VAL.PATH, 
            Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ])
        )
        return DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE//cfg.NUM_GPUS, 
            shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

datamodules = {
    "cifar100": Cifar100DataModule,
    "imagenet": ImageNetDataModule,
}

def build_dataset(cfg):
    return datamodules[cfg.TRAIN.DATASET](cfg)
