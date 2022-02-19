import numpy
import torch
from torch import nn
from torch.autograd import Function
import backward_ops

class DropITer(object):
    def __init__(self, strategy, gamma, d=None):
        self.gamma = gamma
        self.reserve = 1 - gamma
        if d is not None:
            self.d = d
            self.d_reserve = int((1 - gamma) * d)
        self.select = getattr(self, f"select_{strategy}")
        self.pad = getattr(self, f"pad_{strategy}")
    
    # --- RANDOM CHANNEL ---  
    def select_random_c(self, x: torch.Tensor):
        c = x.shape[1]
        idxs = numpy.random.choice(c, int(c * self.reserve))
        x = x[:,idxs]
        x.idxs = idxs
        return x

    def pad_random_c(self, x: torch.Tensor, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x[:,x.idxs] = x
        return _x
    # --- RANDOM CHANNEL ---  

    # --- MINK CHANNEL ---  
    def select_mink_c(self, x: torch.Tensor):
        c = x.shape[1]
        x = x.view(x.shape[0], c, -1)
        idxs = x.norm(p=2, dim=(0,2)).topk(int(c * self.reserve), sorted=False)[1]
        x = x[:,idxs]
        x.idxs = idxs
        return x

    def pad_mink_c(self, x: torch.Tensor, ctx):
        b, c = ctx.input_shape[:2]
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(b, c, -1)[:,x.idxs] = x
        return _x
    # --- MINK CHANNEL ---    

    # --- RANDOM ---    
    def select_random(self, x: torch.Tensor):
        x = x.view(-1)
        idxs = numpy.random.choice(len(x), int(len(x) * self.reserve))
        x = x[idxs]
        x.idxs = idxs
        return x
    
    def pad_random(self, x, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(-1)[x.idxs] = x
        return _x
    # --- RANDOM ---  

    # --- MINK ---  
    def select_mink(self, x: torch.Tensor):
        x, idxs = x.view(-1).topk(int(x.numel() * self.reserve), sorted=False)
        x.idxs = idxs
        return x
    
    def pad_mink(self, x, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(-1).scatter_(0, x.idxs, x)
        return _x
    # --- MINK --- 

    # --- FASTMINK ---  
    def select_fastmink(self, x: torch.Tensor):
        x, idxs = x.view(-1, self.d).topk(self.d_reserve, dim=1, sorted=False)
        x.idxs = idxs
        return x
        
    def pad_fastmink(self, x, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(x.shape[0], -1).scatter_(1, x.idxs, x)
        return _x
    # --- FASTMINK ---  

class _DropITConv2d(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter or None,
        conv2d: nn.Conv2d,
        dropiter: DropITer,
    ):
        ctx.stride = conv2d.stride
        ctx.padding = conv2d.padding
        ctx.dilation = conv2d.dilation
        ctx.groups = conv2d.groups 
        ctx.dropiter = dropiter
        ctx.input_shape = input.shape
        ctx.has_bias = bias is not None
        ctx.save_for_backward(dropiter.select(input), weight)
        return torch.functional.F.conv2d(input, weight, bias, conv2d.stride,
            conv2d.padding, conv2d.dilation, conv2d.groups)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, (0,2,3)) if ctx.has_bias else None
        input = ctx.dropiter.pad(input, ctx)
        grad_input = backward_ops.cudnn_convolution_backward_input(
            ctx.input_shape, grad_output.contiguous(), weight, 
            ctx.padding, ctx.stride, ctx.dilation, ctx.groups, 
            torch.backends.cudnn.benchmark, 
            torch.backends.cudnn.deterministic, 
            torch.backends.cudnn.allow_tf32, 
        )
        grad_weight = backward_ops.cudnn_convolution_backward_weight(
            weight.shape, grad_output.contiguous(), input, 
            ctx.padding, ctx.stride, ctx.dilation, ctx.groups, 
            torch.backends.cudnn.benchmark, 
            torch.backends.cudnn.deterministic, 
            torch.backends.cudnn.allow_tf32, 
        )
        return grad_input, grad_weight, grad_bias, \
            None, None

class DropITConv2d(nn.Module):
    def __init__(self, conv2d: nn.Conv2d, dropiter: DropITer):
        super().__init__()
        self.conv2d = conv2d
        self.dropiter = dropiter
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv2d = self.conv2d
        if self.training: 
            return _DropITConv2d.apply(
                x, conv2d.weight, conv2d.bias,
                conv2d, self.dropiter
            )
        else:
            return conv2d(x)

class _DropITLinear(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter or None,
        dropiter: DropITer,
    ):
        ctx.dropiter = dropiter
        ctx.input_shape = input.shape
        ctx.has_bias = bias is not None
        output = torch.functional.F.linear(input, weight, bias)
        ctx.save_for_backward(dropiter.select(input), weight)
        return output
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, 
            list(range(grad_output.dim()-1))) if ctx.has_bias else None
        input = ctx.dropiter.pad(input, ctx)
        grad_input = grad_output.matmul(weight)
        ic, oc = input.shape[-1], grad_output.shape[-1]
        grad_weight = grad_output.view(-1, oc).T.mm(input.view(-1, ic))
        return grad_input, grad_weight, grad_bias, None

class DropITLinear(nn.Module):
    def __init__(self, linear: nn.Linear, dropiter: DropITer):
        super().__init__()
        self.linear = linear
        self.dropiter = dropiter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.linear
        if self.training:
            return _DropITLinear.apply(
                x, linear.weight, linear.bias,
                self.dropiter
            )
        else:
            return linear(x)

Implemented = {
    'Conv2d': DropITConv2d,
    'Linear': DropITLinear,
}

first_flag = True
def to_dropit(model: nn.Module, dropiter: DropITer):
    for child_name, child in model.named_children():
        type_name = type(child).__name__
        if type_name in Implemented:
            global first_flag
            if first_flag:
                first_flag = False
                continue
            setattr(model, child_name, Implemented[type_name](child, dropiter))
            print(f"{type(child).__name__} -> {Implemented[type_name].__name__}")
        else:
            to_dropit(child, dropiter)
