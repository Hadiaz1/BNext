#!/bin/bash
clear
mkdir log 
python train_assistant_group_amp.py --model bnext_tiny --distillation True --teacher_num 2 --assistant_teacher_num 1 --weak_teacher EfficientNet_B0 --mixup 0.0 --cutmix 0.0 --aug-repeats 1  --batch_size 512 --learning_rate=1e-3  --epochs=512 --weight_decay=0 | tee -a script/BNext-Tiny/log/training.txt


