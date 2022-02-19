import torch
from pytorch_lightning import seed_everything
from torchvision.models import resnet18, resnet50, resnet152
from pytorchvideo.models.hub import x3d_xs
import timm
import time
from layers import to_dropit, DropITer, Contiguous

if __name__ == "__main__":  
    seed_everything(42, workers=True)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # model = resnet18(num_classes=1000)
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100).cuda()
    model.pre_logits = Contiguous()
    x = torch.rand(128,3,224,224).cuda()
    # model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model.maxpool = torch.nn.Identity()
    dropiner = DropITer("fastmink", 0.9, 16384)
    to_dropit(model, dropiner)
    model = model.cuda()
    
    if True:
        print('training')
        model.train()
        start = time.time()
        for i in range(100):
            y = model(x)
            y.sum().backward()
        end = time.time()
    else:
        print('inference')  
        model.eval()
        with torch.no_grad():
            y = model(x)
    print((end - start) / 100 * 1000)
    