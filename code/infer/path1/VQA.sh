#!/bin/bash
imagebind_ckpt_path=./pandagpt/ckpt/imagebind
vicuna_ckpt_path=./pandagpt/ckpt/llama-7b
delta_ckpt_path=./pandagpt/ckpt/VQA.pt
stage=2
max_tgt_len=128
lora_r=32
lora_alpha=32
lora_dropout=0.1
device=cuda:0
orignal_json_path=./pandagpt/data/VQA/val.json
image_dir=./pandagpt/data/VQA/images


python ./pandagpt/VQA.py --imagebind-ckpt-path=$imagebind_ckpt_path \
    --vicuna-ckpt-path=$vicuna_ckpt_path \
    --delta-ckpt-path=$delta_ckpt_path --stage=$stage  --device=$device \
    --max-tgt-len=$max_tgt_len  --lora-r=$lora_r  --lora-alpha=$lora_alpha \
    --lora-dropout=$lora_dropout  --orignal-json-path=$orignal_json_path  \
    --image-dir=$image_dir  