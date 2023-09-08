from layer_utils import CovpoolLayer
from utils_quant import *

import torch
import torch.nn.functional as F
import nni
import nni.retiarii.nn.pytorch as nn
import math
from nni.retiarii import model_wrapper



from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.evaluator import FunctionalEvaluator
import nni.retiarii.strategy as strategy

from timm.scheduler.cosine_lr import CosineLRScheduler
import pytorch_warmup as warmup
import argparse
import warnings
import numpy as np
import time, datetime
import logging

from torchvision import datasets, transforms

from utils.utils import AverageMeter, ProgressMeter, accuracy, compute_params_ROM_MACs

import yaml

with open("params.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)
 
def adjust_temperature(model, epoch):
    temperature = torch.ones(1)
    from birealnet import HardSign, HardBinaryConv
    for module in model.modules():
        if isinstance(module, (HardSign, HardBinaryConv)):
            if (epoch % 1)==0 and (epoch != 0):
                module.temperature.mul_(0.9)
            temperature = module.temperature
    return temperature


class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride, quant = True):
        super(firstconv3x3, self).__init__()
        if not quant:
            self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        else:
            self.conv1 = QuantizeConv(inp, oup, 3, stride, 1, 1, 1, False, activation_bits = 8, weight_bits = 8) 

        self.bn1 = nn.BatchNorm2d(oup)
        self.prelu = nn.PReLU(oup, oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        return out


def conv3x3(in_planes, out_planes, kernel_size = 3, stride=1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, dilation = dilation, groups = groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding = 0, bias=False)

class HardSign(nn.Module):
    def __init__(self, range=[-1, 1], progressive=False):
        super(HardSign, self).__init__()
        self.range = range
        self.progressive = progressive
        self.register_buffer("temperature", torch.ones(1))

    def adjust(self, x, scale=0.1):
        self.temperature.mul_(scale)

    def forward(self, x, scale=None):
        if scale == None:
            scale = torch.ones_like(x)

        replace = x.clamp(self.range[0], self.range[1]) + scale
        x = x.div(self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        if not self.progressive:
            sign = x.sign() * scale
        else:
            sign = x * scale
        return (sign - replace).detach() + replace

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.prelu = nn.PReLU(oup, oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.number_of_weights = in_chn // groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn((self.shape)) * 0.001, requires_grad=True)
        # self.weight_bias = nn.Parameter(torch.zeros(out_chn, in_chn, 1, 1))
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        self.weight.data.clamp_(-1.5, 1.5)
        real_weights = self.weight

        binary_weights_no_grad = (real_weights / self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        if self.temperature < 1e-5:
            binary_weights_no_grad = binary_weights_no_grad.sign()
        cliped_weights = real_weights
        if self.training:
            binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            binary_weights = binary_weights_no_grad

        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y

# class SqueezeAndExpand(nn.Module):
#     def __init__(self, channels, planes, ratio=8, attention_mode="hard_sigmoid"):
#         super(SqueezeAndExpand, self).__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(channels, channels // ratio, kernel_size=1, padding=0),
#             nn.ReLU(channels // ratio),
#             nn.Conv2d(channels // ratio, planes, kernel_size=1, padding=0),
#         )

#         if attention_mode == "sigmoid":
#             self.attention = nn.Sigmoid()

#         elif attention_mode == "hard_sigmoid":
#             self.attention = HardSigmoid()

#         else:
#             self.attention = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.se(x)
#         x = self.attention(x)
#         return x

class SqueezeAndExpand(nn.Module):
    def __init__(self, channels, planes, ratio=8, attention_mode="hard_sigmoid", quant = True, bits = 8):
        super(SqueezeAndExpand, self).__init__()
        if not quant:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(channels, channels // ratio, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels // ratio, planes, kernel_size=1, padding=0),
            )

        else:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                QuantizeConv(channels, channels // ratio, 1, 1, 0, 1, 1, True, activation_bits=bits, weight_bits=bits),
                nn.ReLU(),
                QuantizeConv(channels // ratio, channels, 1, 1, 0, 1, 1, True, activation_bits=bits, weight_bits=bits)

            )

        if attention_mode == "sigmoid":
            self.attention = nn.Sigmoid()

        elif attention_mode == "hard_sigmoid":
            self.attention = HardSigmoid()

        else:
            self.attention = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.se(x)
        x = self.attention(x)
        return x

class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        self.t = int(abs((math.log(channels, 2) + b) / gamma))
        self.k = self.t if self.t % 2 else self.t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k, padding=int(self.k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x.squeeze(-1).transpose(-1, -2))
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = self.sigmoid(x)
        return x

class GSoPAttention(nn.Module):
    def __init__(self, channels, att_dim=16, quant=True, bits=8):
        super(GSoPAttention, self).__init__()
        if channels > 64:
            DR_stride = 1
        else:
            DR_stride = 2

        self.dimDR = att_dim

        if not quant:
            self.gsop_conv = nn.Conv2d(channels, self.dimDR, kernel_size=1, stride=DR_stride, bias=False)
            self.gsop_fc = nn.Conv2d(4*self.dimDR, channels, kernel_size=1, groups=1, bias=False)
        
        else:
            self.gsop_conv = QuantizeConv(channels, self.dimDR, 1, DR_stride, 0, 1, 1, False, activation_bits=bits, weight_bits=bits)
            self.gsop_fc = QuantizeConv(4*self.dimDR, channels, 1, 1, 0, 1, 1, False, activation_bits=bits, weight_bits=bits)
        
        self.row_conv = nn.Conv2d(self.dimDR, 4*self.dimDR, kernel_size=(self.dimDR, 1), groups=self.dimDR, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dimDR)
        self.relu1 = nn.ReLU()
        self.row_bn = nn.BatchNorm2d(self.dimDR)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gsop_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = CovpoolLayer(x)
        x = x.view(x.size(0), x.size(1), x.size(2), 1).contiguous()
        x = self.row_bn(x)

        x = self.row_conv(x)

        x = self.gsop_fc(x)
        x = self.sigmoid(x)

        return x

class Attention(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, gamma=1e-6, groups=1, att_module_elm="SE", att_in_elm="pre_post", activation="prelu"):
        super(Attention, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.att_module = att_module_elm
        self.att_in = att_in_elm

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        if activation == "prelu":
            self.activation1 = nn.PReLU(inplanes)
            self.activation2 = nn.PReLU(planes)
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.downsample = downsample
        self.stride = stride
        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)


        self.se = att_module_elm

        self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):
        if self.training:
            self.scale.data.clamp_(0, 1)

        residual = self.activation1(input)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)

        if self.att_in == "pre_post":
            inp = self.scale * residual + x * (1 - self.scale)
        elif self.att_in == "post":
            inp = x
        elif self.att_in == "pre":
            inp = residual
        else:
            raise ValueError("This Attention Block is not implemented")

        x = self.se(inp) * x

        x = x * residual
        x = self.norm2(x)
        x = x + residual

        return x

class FFN_3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, gamma=1e-8, groups=1, att_module_ffn="SE", att_in_ffn="pre_post", activation="prelu"):
        super(FFN_3x3, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.att_module = att_module_ffn
        self.att_in = att_in_ffn

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        if activation == "prelu":
            self.activation1 = nn.PReLU(inplanes)
            self.activation2 = nn.PReLU(planes)
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.downsample = downsample

        self.se = att_module_ffn

        self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):
        self.scale.data.clamp_(0, 1)

        residual = input

        if self.stride == 2:
            residual = self.downsample(residual)

        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)

        if self.att_in == "pre_post":
            inp = self.scale * residual + x * (1 - self.scale)
        elif self.att_in == "post":
            inp = x
        elif self.att_in == "pre":
            inp = residual
        else:
            raise ValueError("This Attention Block is not implemented")

        x = self.se(inp) * x

        x = self.norm2(x)
        x = x + residual

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate = 0.1, mode = "scale", att_module_elm="SE", att_in_elm="pre_post", att_module_ffn=None, att_in_ffn=None, activation="prelu"):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        
        self.Attention = Attention(inplanes, planes, stride, downsample, drop_rate = drop_rate, groups = 1, att_module_elm=att_module_elm, att_in_elm=att_in_elm, activation=activation)
         
        self.FFN  = FFN_3x3(planes, planes, 1, None, drop_rate = drop_rate, groups = 1, att_module_ffn=att_module_ffn, att_in_ffn=att_in_ffn, activation=activation)
          
    def forward(self, input):
        x = self.Attention(input)
        y = self.FFN(x)

        return y
      

@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self, num_classes=10, block=BasicBlock, n_layers = 3, activation="prelu"):
        super().__init__()
        width = 1
        self.num_layers = n_layers
        self.num_blocks_per_layer = [2] * self.num_layers
        self.inplanes = nn.ModelParameterChoice([16, 32, 64], label="inplanes")

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = QuantizeConv(3, self.inplanes, 3, 1, 1, 1, 1, False, activation_bits = 8, weight_bits = 8) 
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = lambda x:x

        width_base_multiplier = self.inplanes
        width_multiplier = [width_base_multiplier*(2**i) for i in range(self.num_layers)]

        self.layers = torch.nn.ModuleList()
        self.atts = []

        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(self._make_layer(block, int(width*width_multiplier[i]), self.num_blocks_per_layer[i], att_module_elm_1=nn.LayerChoice([SqueezeAndExpand(int(width*width_multiplier[i]), int(width*width_multiplier[i]), attention_mode="sigmoid")
                ,EfficientChannelAttention(int(width*width_multiplier[i])),GSoPAttention(int(width*width_multiplier[i]))], label=f"att_elm_1_{i}"), att_in_elm_1=nn.ModelParameterChoice(["pre", "post", "pre_post"], label=f"att_in_elm_1_{i}"),att_module_ffn_1=nn.LayerChoice([SqueezeAndExpand(int(width*width_multiplier[i]), int(width*width_multiplier[i]), attention_mode="sigmoid")
                ,EfficientChannelAttention(int(width*width_multiplier[i])),GSoPAttention(int(width*width_multiplier[i]))], label=f"att_ffn_1_{i}"), att_in_ffn_1=nn.ModelParameterChoice(["pre", "post", "pre_post"], label=f"att_in_ffn_1_{i}"), att_module_elm_2=nn.LayerChoice([SqueezeAndExpand(int(width*width_multiplier[i]), int(width*width_multiplier[i]), attention_mode="sigmoid")
                ,EfficientChannelAttention(int(width*width_multiplier[i])),GSoPAttention(int(width*width_multiplier[i]))], label=f"att_elm_2_{i}"), att_in_elm_2=nn.ModelParameterChoice(["pre", "post", "pre_post"], label=f"att_in_elm_2_{i}"),att_module_ffn_2=nn.LayerChoice([SqueezeAndExpand(int(width*width_multiplier[i]), int(width*width_multiplier[i]), attention_mode="sigmoid")
                ,EfficientChannelAttention(int(width*width_multiplier[i])),GSoPAttention(int(width*width_multiplier[i]))], label=f"att_ffn_2_{i}"), att_in_ffn_2=nn.ModelParameterChoice(["pre", "post", "pre_post"], label=f"att_in_ffn_2_{i}"),activation=activation))
            else:
                self.layers.append(self._make_layer(block, int(width*width_multiplier[i]), self.num_blocks_per_layer[i], stride=2, att_module_elm_1=nn.LayerChoice([SqueezeAndExpand(int(width*width_multiplier[i]), int(width*width_multiplier[i]), attention_mode="sigmoid")
                ,EfficientChannelAttention(int(width*width_multiplier[i])),GSoPAttention(int(width*width_multiplier[i]))], label=f"att_elm_1_{i}"), att_in_elm_1=nn.ModelParameterChoice(["pre", "post", "pre_post"], label=f"att_in_elm_1_{i}"),att_module_ffn_1=nn.LayerChoice([SqueezeAndExpand(int(width*width_multiplier[i]), int(width*width_multiplier[i]), attention_mode="sigmoid")
                ,EfficientChannelAttention(int(width*width_multiplier[i])),GSoPAttention(int(width*width_multiplier[i]))], label=f"att_ffn_1_{i}"), att_in_ffn_1=nn.ModelParameterChoice(["pre", "post", "pre_post"], label=f"att_in_ffn_1_{i}"), att_module_elm_2=nn.LayerChoice([SqueezeAndExpand(int(width*width_multiplier[i]), int(width*width_multiplier[i]), attention_mode="sigmoid")
                ,EfficientChannelAttention(int(width*width_multiplier[i])),GSoPAttention(int(width*width_multiplier[i]))], label=f"att_elm_2_{i}"), att_in_elm_2=nn.ModelParameterChoice(["pre", "post", "pre_post"], label=f"att_in_elm_2_{i}"),att_module_ffn_2=nn.LayerChoice([SqueezeAndExpand(int(width*width_multiplier[i]), int(width*width_multiplier[i]), attention_mode="sigmoid")
                ,EfficientChannelAttention(int(width*width_multiplier[i])),GSoPAttention(int(width*width_multiplier[i]))], label=f"att_ffn_2_{i}"), att_in_ffn_2=nn.ModelParameterChoice(["pre", "post", "pre_post"], label=f"att_in_ffn_2_{i}"), activation=activation))
      
        if activation == "prelu":
            self.prelu = nn.PReLU(width_multiplier[-1])
        elif activation == "gelu":
            self.prelu = nn.GELU()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(width_multiplier[-1], num_classes)
        self.fc = QuantizeLinear(width_multiplier[-1], num_classes, activation_bits = 8, weight_bits = 8)
    

    def _make_layer(self, block, planes, blocks, stride=1, att_module_elm_1=None, att_module_ffn_1=None, att_in_elm_1="pre_post", att_in_ffn_1="pre_post", att_module_elm_2=None, att_module_ffn_2=None, att_in_elm_2="pre_post", att_in_ffn_2="pre_post", activation="prelu"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                                nn.AvgPool2d(kernel_size=2, stride=stride),
                                conv1x1(self.inplanes, planes * block.expansion),
                                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate = 0, att_module_elm=att_module_elm_1, att_module_ffn=att_module_ffn_1, att_in_elm=att_in_elm_1, att_in_ffn=att_in_ffn_1, activation=activation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_rate = 0, att_module_elm=att_module_elm_2, att_module_ffn=att_module_ffn_2, att_in_elm=att_in_elm_2, att_in_ffn=att_in_ffn_2, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        for i in range(self.num_layers):
            x = self.layers[i].cuda()(x)

        x = self.prelu(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def train_epoch(model, device, train_loader, criterion, scheduler, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_all = AverageMeter('Loss_ALL', ':.4e')
    losses_entropy = AverageMeter('Loss_Entropy', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_all, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    scheduler.step(epoch = epoch)
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, batch in enumerate(train_loader):
        images = batch[0].to(device)
        target = batch[1].to(device)

        logits = model(images)

        loss = criterion(logits, target)
        alpha = torch.zeros(1)
        loss_all = loss

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))

        n = images.size(0)

        losses_all.update(loss_all.item(), n)   #accumulated loss
        losses_entropy.update(loss.item(), n)

        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        loss_all.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        progress.display(i)


def test_epoch(model, device, criterion, val_loader):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch[0].to(device)
            target = batch[1].to(device)

            logits = model(images)
            loss = criterion(logits, target)

            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0].item(), n)
            top5.update(pred5[0].item(), n)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            progress.display(i)

    return top1.avg


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    training_temperature = []
    criterion = nn.CrossEntropyLoss().cuda()
    all_parameters = model.parameters()
    weight_parameters = []

    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or 'binary_conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    
    attention_parameters = []
    for pname, p in model.named_parameters():
        if 'binary_conv' in pname:
            attention_parameters.append(p)
    
    normal_parameters = []
    normal_parameters = list(filter(lambda p: id(p) not in (weight_parameters_id + weight_parameters_id), all_parameters))
    
    optimizer = torch.optim.AdamW(
            [{'params' : other_parameters, 'weight_decay': 1e-3},
             {'params' : weight_parameters, 'weight_decay' : 1e-8}],
            lr=CONFIG["learning_rate"])


    scheduler = CosineLRScheduler(optimizer, t_initial=CONFIG["epochs"], warmup_t=5, lr_min = 1e-7, warmup_lr_init=1e-5, warmup_prefix = True)
    
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std = [0.267, 0.256, 0.276])
        
    #train data arguments
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.CIFAR10(root = "datasets/CIFAR10/", train = True, download = True, transform = train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)
        
    #val data arguments
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    val_dataset = datasets.CIFAR10(root = "datasets/CIFAR10/", train = False, download = True, transform = val_transforms)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=True)


    temperature = 1.0
    training_temperature.append(temperature)

    best_top1_acc = 0

    params, footprint, macs = compute_params_ROM_MACs(model, ptq=8)

    for epoch in range(CONFIG["epochs"]):
        # train the model for one epoch
        train_epoch(model, device, train_loader, criterion, scheduler, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, criterion, val_loader)

        is_best = False
        if accuracy > best_top1_acc:
            best_top1_acc = accuracy
            is_best = True


        
        temperature = adjust_temperature(model, epoch).item()
        training_temperature.append(temperature)
        print("Best Acc: {}%, Temp: {}".format(best_top1_acc, temperature))

        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result({"accuracy": accuracy})


    # report final test result
    nni.report_final_result({"default": accuracy, "memory footprint kib": footprint, "MACs": macs, "number of params": params})



if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="run",
                    help='NAS mode: "run" or "resume" or "view". "resume" and "view" require the experiment ID.')
    parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of Layers for all the candidate models to search for')
    parser.add_argument('--experiment_name', type=str, default="CIFAR10_search",
                    help='Name assigned to the experiment')
    parser.add_argument('--strategy', type=str, default="random",
                    help='search strategy: "reg_evo" or "random"')
    parser.add_argument('--num_trials', type=int, default=5,
                    help='Number of Models to be evaluated in the search')
    parser.add_argument('--port', type=int, default=8083,
                    help='Port Number')
    parser.add_argument('--experiment_id', type=str, default=None,
                    help='Experiment ID of the experiment to be resumed ir viewed. Will be automatically set when running a new experiment for first time')
    
    args = parser.parse_args()
    model_space = ModelSpace(n_layers=args.num_layers)

    evaluator = FunctionalEvaluator(evaluate_model)

    if args.strategy == "reg_evo":
        search_strategy = strategy.RegularizedEvolution()
    elif args.strategy == "random":
        search_strategy = strategy.Random(dedup=True)
    else:
        raise ValueError("The chosen strategy is not supported." 
        "Please Choose one of the following: 'reg_evo' - 'random' ")

    exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 2  
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = True

    
    if args.mode == "run":
        exp_config.experiment_name = args.experiment_name
        exp_config.max_trial_number = args.num_trials  
        exp.run(exp_config, args.port)

    elif args.mode == "resume":
        exp.resume(args.experiment_id, port=args.port)

    elif args.mode == "view":
        exp.view(args.experiment_id, port=args.port)
    
    else:
        raise ValueError("The chosen mode is not supported."
        "Please Choose one of the following: 'run' - 'resume' - 'view'")


    for model_dict in exp.export_top_models(formatter='dict'):
        print(model_dict)
        with nni.retiarii.fixed_arch(model_dict):
            final_model = ModelSpace()
            print(final_model)
