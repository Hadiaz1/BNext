### EXAMPLE: Converting Bnex18 quantized memory footprint 131.6 KiB accuracy = 90.14% with 2 stages

from birealnet_quant import *
from utils.utils import compute_params_ROM, torch_to_onnx_converter
model = birealnet18(num_classes=10, quant=8)
example_input = torch.randn(1, 3, 32, 32)
model(example_input)
compute_params_ROM(model, ptq=8)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)

checkpoint = r"C:\Users\alzeinha\BNext\src\models\CIFAR100_bnext18_quant_optimizer_AdamW_mixup_0.0_cutmix_0.0_aug_repeats_1_KD_0_assistant_0_ResNet101_HK_False_Instance_aa_rand-m7-mstd0.5-inc1__elm_True_recoup_True_None_amp_2layers_QAT\model_best.pth.tar"



torch_to_onnx_converter(torch_model=model,
                        device=device,
                        ckpt_path=checkpoint,
                        input_shape=(1, 3, 32, 32))