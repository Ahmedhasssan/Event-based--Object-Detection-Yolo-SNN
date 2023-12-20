import torch
import torch.nn as nn
import torch.nn.functional as F
#from surrogate import *
from torch import Tensor
import math
from models.q_modules import *

def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(2):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 3):  ### there is 3 instead of 2 for original implementations of 4 bit
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values

def act_quantization(b, grid, power=True):

    def power_quant(x, grid):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        ## For flooring
        #d = xhard.unsqueeze(0) - value_s.unsqueeze(1)
        #d[d.lt(0.)]=1.0
        #idxs = d.abs().min(dim=0)[1]
        ################
        xhard = value_s[idxs].view(shape)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha=2):
            input_c = input.clamp(min=-2,max=1)
            #grid = torch.Tensor([-2.0,-1.8750,-1.750,-1.6250,-1.50,-1.3750,-1.250,-1.1250,-1.00, -0.8750,
            #-0.750,-0.6250,-0.50,-0.3750,-0.250,-0.125,0,0.125,0.3750,0.50,0.6250,0.750,0.8750])
            grid = torch.Tensor([-2.0,-1.0,0.0,1.0])
            input_q = power_quant(input_c, grid)
            ctx.save_for_backward(input, input_q)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_input = grad_input*(1-i)
            return grad_input, None

    return _uq().apply

act_grid = build_power_value(4, additive=False)
log2 = act_quantization(4, act_grid, power=True)


class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.mask = None
        self.p = p

    def extra_repr(self):
        return 'p={}'.format(
            self.p
        )

    def create_mask(self, x: torch.Tensor):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x: torch.Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return x * self.mask
        else:
            return x

    def reset(self):
        self.mask = None


class Layer_LP(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, wbit):
        super(Layer_LP, self).__init__()
        self.fwd = SeqToANNContainer(
            QConv2d(in_plane, out_plane, kernel_size, stride=stride, padding=padding, bias=True, wbit=wbit, abit=32, infer=False),
            nn.BatchNorm2d(out_plane)
        )
        #self.act = LIFSpike()
        self.act=ZIFArchTan()
    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class Dropout2d(Dropout):
    def __init__(self, p=0.2):
        super().__init__(p)

    def create_mask(self, x: torch.Tensor):
        self.mask = F.dropout2d(torch.ones_like(x.data), self.p, training=True)
    def forward(self, x: torch.Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return x * self.mask
        else:
            return x

class SConv(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(SConv, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        #self.act = LIFSpike()
        self.act=ZIFArchTan()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class SConvDW(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(SConvDW, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,in_plane,kernel_size,stride,padding),
            nn.Conv2d(in_plane,out_plane,1,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        #self.act = LIFSpike()
        self.act=ZIFArchTan()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        #self.act = LIFSpike()
        self.act=ZIFArchTan()
        
    def forward(self,x):
        x=x
        x = self.fwd(x)
        x = self.act(x)
        return x

class Linear(nn.Module):
    def __init__(self,in_plane,out_plane):
        super(Linear, self).__init__()
        self.L1 = SeqToANNContainer(
            nn.Linear(in_plane,out_plane),
        )
        self.act = LIFSpike()
        #self.act=LIFSpike()

    def forward(self,x):
        x = self.L1(x)
        x = self.act(x)
        return x

class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = LIFSpike()
        #self.act=ZIFArchTan()
    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

######### Surrogate Gradient ATAN ############
def heaviside(x:Tensor):
    return x.ge(0.).float()

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gama, salpha, thresh):
        #if x.requires_grad:
            out= heaviside(x)
            ctx.alpha = thresh
            L = torch.tensor([gama])
            ctx.save_for_backward(x, out, L, thresh)
            #out=out.mul(salpha.half())
            return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        grad_alpha = None
        grad_input = grad_output.clone()
        if ctx.needs_input_grad[0]:
            grad_input = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output
        (input, out, others, thresh) = ctx.saved_tensors
        gama = others[0].item()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None, None, None

######### Surrogate Gradient Sigmoid ############
def heaviside(x:Tensor):
    return x.ge(0.).float()

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gama, salpha, thresh):
            out= heaviside(x)
            ctx.alpha = thresh
            L = torch.tensor([gama])
            ctx.save_for_backward(x, out, L, thresh)
            return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others, thresh) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None, None, None

class ZIFArchTan(nn.Module):
    r"""
    Arch Tan function
    """
    def __init__(self, thresh=1.0, tau=1, gama=1.0):
        super(ZIFArchTan, self).__init__()
        self.act = sigmoid.apply
        self.thresh = thresh
        self.act_alpha = 2
        self.tau = tau
        self.neuron_idx = 0
        self.firing_rate = 0
        self.gama = gama
        self.thresh = nn.Parameter(torch.Tensor([thresh]), requires_grad=True)
        self.salpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            vth = self.thresh
            spike = self.act(mem - vth, self.gama, self.salpha, self.thresh)
            mem = (1 - spike) * mem
            #### Low Precision of Membrane Potential #####
            #mem = log2(mem, self.act_alpha)
            ###### Save Membrane Potential ######
            print(self.neuron_idx)
            pot = mem.detach().cpu()
            torch.save(pot, f"./vmem/neuron{self.neuron_idx}_t{t}.pt")
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)

############################################

class LVZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama, thresh):
        #input = input-thresh
        out = input.gt(thresh).float().mul(3.3*thresh)
        #out = (input-thresh > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L, thresh)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others, thresh) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        mask_alpha = input.ge(thresh[0].item()).float()
        grad_alpha = torch.sum(grad_output.mul(mask_alpha)).view(-1)
        return grad_input, None, grad_alpha

class SLSZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama, salpha, thresh):
        #input = input-thresh
        out = input.gt(thresh).float()
        #out = (input-thresh > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        out=out.mul(salpha.half())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        s = ctx.saved_tensors[1]
        grad_alpha = torch.sum(grad_output.mul(s)).view(-1)
        return grad_input, None, grad_alpha, None


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y


# LIFSpike = LIF
