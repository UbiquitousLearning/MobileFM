#!/bin/bash
imagebind_ckpt_path=/workspace/model/imagebind
whisper_ckpt_path=/workspace/model/whisper_ckpt/tiny-en
vicuna_ckpt_path=/workspace/model/llama-7b
delta_ckpt_path=./pandagpt/ckpt/FSC.pt
stage=2
max_tgt_len=128
lora_r=32
lora_alpha=32
lora_dropout=0.1
device=cuda:0
orignal_csv_path=./pandagpt/data/FSC/test-clean.csv
image_dir=./pandagpt/data/FSC

python ./pandagpt/FSC.py --imagebind-ckpt-path=$imagebind_ckpt_path \
    --whisper-ckpt-path=$whisper_ckpt_path  --vicuna-ckpt-path=$vicuna_ckpt_path \
    --delta-ckpt-path=$delta_ckpt_path --stage=$stage  --device=$device \
    --max-tgt-len=$max_tgt_len  --lora-r=$lora_r  --lora-alpha=$lora_alpha \
    --lora-dropout=$lora_dropout  --orignal-csv-path=$orignal_csv_path  --image-dir=$image_dir