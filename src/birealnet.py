import torch
import torch.nn as nn
from torchinfo import summary
import torchvision
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
from layer_utils import *
import bam
import cbam

def conv3x3(in_planes, out_planes, kernel_size = 3, stride=1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, dilation = dilation, groups = groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding = 0, bias=False)

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

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


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

class SqueezeAndExpand(nn.Module):
    def __init__(self, channels, planes, ratio=8, attention_mode="hard_sigmoid"):
        super(SqueezeAndExpand, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // ratio, kernel_size=1, padding=0),
            nn.ReLU(channels // ratio),
            nn.Conv2d(channels // ratio, planes, kernel_size=1, padding=0),
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

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.chanel_in = channels

        self.query = nn.Conv2d(in_channels=channels,
                             out_channels=channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels=channels,
                           out_channels=channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels=channels,
                             out_channels=channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query(x).reshape(
            m_batchsize, -1, width*height).transpose(1, 2)
        proj_key = self.key(x).reshape(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).reshape(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.transpose(1, 2))
        out = out.reshape(m_batchsize, C, height, width)

        return out

class GSoPAttention(nn.Module):
    def __init__(self, channels, att_dim=128):
        super(GSoPAttention, self).__init__()
        if channels > 64:
            DR_stride = 1
        else:
            DR_stride = 2

        self.dimDR = att_dim
        self.conv1 = nn.Conv2d(channels, self.dimDR, kernel_size=1, stride=DR_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dimDR)
        self.relu1 = nn.ReLU()

        self.row_bn = nn.BatchNorm2d(self.dimDR)
        self.row_conv = nn.Conv2d(self.dimDR, 4*self.dimDR, kernel_size=(self.dimDR, 1), groups=self.dimDR, bias=False)
        self.fc = nn.Conv2d(4*self.dimDR, channels, kernel_size=1, groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = CovpoolLayer(x)
        x = x.view(x.size(0), x.size(1), x.size(2), 1).contiguous()
        x = self.row_bn(x)

        x = self.row_conv(x)

        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class BAM(nn.Module):
    def __init__(self, channels):
        super(BAM, self).__init__()
        self.channel_att = bam.ChannelGate(channels)
        self.spatial_att = bam.SpatialGate(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = 1 + self.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return x

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = cbam.ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = cbam.SpatialGate()

    def forward(self, x):
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        return x


class CGDAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, nonlinear=True):
        super(CGDAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.w0 = nn.Parameter(torch.ones(in_channels, 1), requires_grad=True)
        self.w1 = nn.Parameter(torch.ones(in_channels, 1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(in_channels, 1), requires_grad=True)

        self.bias0 = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)

        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        # self.tanh = nn.Tanh()

    def forward(self, x):
        b, c, _, _ = x.size()
        x0 = self.avg_pool(x).view(b, c, 1, 1)
        x1 = self.max_pool(x).view(b, c, 1, 1)

        x0_s = self.softmax(x0)  # b ,c ,1 ,1

        y0 = torch.matmul(x0.view(b, c), self.w0).view(b, 1, 1, 1)
        y1 = torch.matmul(x1.view(b, c), self.w1).view(b, 1, 1, 1)

        y0_s = torch.tanh(y0 * x0_s + self.bias0)  # b ,c ,1 ,1
        y1_s = torch.tanh(y1 * x0_s + self.bias1)  # b ,c ,1 ,1

        y2 = torch.matmul(y1_s.view(b, c), self.w2).view(b, 1, 1, 1)
        y2_s = torch.tanh(y2 * y0_s + self.bias2).view(b, c, 1, 1)

        z = x * (y2_s + 1)

        return z

class CGD(nn.Module):
    def __init__(self, in_channels, spatial_size):
        super(CGD, self).__init__()
        self.att_layer = CGDAttentionLayer(spatial_size, spatial_size)

    def forward(self, x):
        x = self.att_layer(x)
        return x

class Attention(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, gamma=1e-6, groups=1, att_module="SE", att_in="pre_post", activation="prelu"):
        super(Attention, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.att_module = att_module
        self.att_in = att_in

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


        if self.att_module == "SE":
            self.se = SqueezeAndExpand(planes, planes, attention_mode="sigmoid")
        elif self.att_module == "ECA":
            self.se = EfficientChannelAttention(planes)
        elif self.att_module.lower() == "GSoP".lower():
            self.se = GSoPAttention(planes)
        elif self.att_module.lower() == "SA".lower():
            self.se = SelfAttention(planes)
        elif self.att_module.lower() == "BAM".lower():
            self.se = BAM(planes)
        elif self.att_module.lower() == "CBAM".lower():
            self.se = CBAM(planes)
        elif self.att_module.lower() == "CGD".lower():
            self.se = CGD(planes, planes)
        else:
            raise ValueError("This Attention Block is not implemented")

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

        if self.att_module == "SA" or self.att_module == "CBAM" or self.att_module == "CGD":
            x = self.se(inp)
        elif self.att_module.lower() == "BAM".lower():
            res = inp
            x = self.se(inp) * inp
            x = res + x
        else:
            x = self.se(inp) * x

        x = x * residual
        x = self.norm2(x)
        x = x + residual

        return x


class FFN_3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, gamma=1e-8, groups=1, att_module="SE", att_in="pre_post", activation="prelu"):
        super(FFN_3x3, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.att_module = att_module
        self.att_in = att_in

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

        if self.att_module == "SE":
            self.se = SqueezeAndExpand(planes, planes, attention_mode="sigmoid")
        elif self.att_module == "ECA":
            self.se = EfficientChannelAttention(planes)
        elif self.att_module.lower() == "GSoP".lower():
            self.se = GSoPAttention(planes)
        elif self.att_module.lower() == "SA".lower():
            self.se = SelfAttention(planes)
        elif self.att_module.lower() == "BAM".lower():
            self.se = BAM(planes)
        elif self.att_module.lower() == "CBAM".lower():
            self.se = CBAM(planes)
        elif self.att_module.lower() == "CGD".lower():
            self.se = CGD(planes, planes)
        else:
            raise ValueError("This Attention Block is not implemented")

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

        if self.att_module == "SA" or self.att_module == "CBAM" or self.att_module == "CGD":
            x = self.se(inp)
        elif self.att_module.lower() == "BAM".lower():
            res = inp
            x = self.se(inp) * inp
            x = res + x
        else:
            x = self.se(inp) * x

        x = self.norm2(x)
        x = x + residual

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate = 0.1, mode = "scale", att_module="SE", att_in="pre_post", activation="prelu"):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        
        self.Attention = Attention(inplanes, planes, stride, downsample, drop_rate = drop_rate, groups = 1, att_module=att_module, att_in=att_in, activation=activation)
         
        self.FFN  = FFN_3x3(planes, planes, 1, None, drop_rate = drop_rate, groups = 1, att_module=att_module, att_in=att_in, activation=activation)
          
    def forward(self, input):
        x = self.Attention(input)
        y = self.FFN(x)

        return y
      

class BNext18(nn.Module):
    def __init__(self, num_classes=1000, block=BasicBlock, layers = [2, 2, 2, 2], att_module="ECA", att_in="pre_post", activation="prelu"):
        super(BNext18, self).__init__()
        drop_rate = 0.2 if num_classes == 100 else 0.
        width = 1
        self.inplanes = 64

        if num_classes == 1000:
            self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = lambda x:x

        self.layer0 = self._make_layer(block, int(width*64), layers[0], att_module=att_module, att_in=att_in, activation=activation)
        self.layer1 = self._make_layer(block, int(width*128), layers[1], stride=2, att_module=att_module, att_in=att_in, activation=activation)
        self.layer2 = self._make_layer(block, int(width*256), layers[2], stride=2, att_module=att_module, att_in=att_in, activation=activation)
        self.layer3 = self._make_layer(block, int(width*512), layers[3], stride=2, att_module=att_module, att_in=att_in, activation=activation)

        if activation == "prelu":
            self.prelu = nn.PReLU(512)
        elif activation == "gelu":
            self.prelu = nn.GELU()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, att_module="ECA", att_in="pre_post", activation="prelu"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                                nn.AvgPool2d(kernel_size=2, stride=stride),
                                conv1x1(self.inplanes, planes * block.expansion),
                                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate = 0, att_module=att_module, att_in=att_in, activation=activation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_rate = 0, att_module=att_module, att_in=att_in, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.prelu(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



if __name__ == "__main__":
    input = torch.randn(1, 3, 224, 224).cuda()
    model = BNext18(num_classes = 1000).cuda()
    print(model(input).size())
    #summary(model, input_size=(1, 3, 224, 224))
    
