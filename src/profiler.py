from birealnet import *
from birealnet_quant import *
from torchinfo import summary
import torch
from prettytable import PrettyTable
from utils.utils import compute_params_ROM


quant = 8

model = birealnet18(num_classes=10, quant=quant)
example_input = torch.randn(1, 3, 32, 32)
model(example_input)



summary(model, input_size=(1, 3, 32, 32))
compute_params_ROM(model, ptq=quant)


