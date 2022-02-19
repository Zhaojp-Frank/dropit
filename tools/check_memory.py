import torch
from pytorch_lightning import seed_everything
from torchvision.models import resnet18, resnet50, resnet152
import timm
from layers import to_dropit, DropITer, Contiguous

if __name__ == "__main__":  
    seed_everything(42, workers=True)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    x = torch.rand(128, 3, 32, 32).cuda()
    # model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)
    # model.pre_logits = Contiguous()
    model = resnet18(num_classes=100)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = torch.nn.Identity()
    dropiter = DropITer("mink_c", 0.7, 16384)
    # to_dropit(model, dropiter)
    model = model.cuda()

    if True:
        print('training')
        model.train()
        y = model(x)
    else:
        print('inference')  
        model.eval()
        with torch.no_grad():
            y = model(x)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(torch.cuda.max_memory_allocated()/1024**2)
    print(torch.cuda.memory_reserved()/1024**2)