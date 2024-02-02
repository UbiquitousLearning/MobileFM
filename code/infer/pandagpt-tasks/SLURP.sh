#!/bin/bash
imagebind_ckpt_path=./pandagpt/ckpt/imagebind
whisper_ckpt_path=./pandagpt/ckpt/whisper_ckpt/tiny-en
vicuna_ckpt_path=./pandagpt/ckpt/llama-7b
delta_ckpt_path=./pandagpt/ckpt/SLURP.pt
stage=2
max_tgt_len=128
lora_r=32
lora_alpha=32
lora_dropout=0.1
device=cuda:0
orignal_csv_path=./pandagpt/data/SLURP/test.csv
image_dir=./pandagpt/data/SLURP
compare_result=./pandagpt/data/SLURP/test.jsonl
our_result=./pandagpt/data/SLURP/prediction.jsonl

python ./pandagpt/SLURP.py --imagebind-ckpt-path=$imagebind_ckpt_path \
    --whisper-ckpt-path=$whisper_ckpt_path  --vicuna-ckpt-path=$vicuna_ckpt_path \
    --delta-ckpt-path=$delta_ckpt_path --stage=$stage  --device=$device \
    --max-tgt-len=$max_tgt_len  --lora-r=$lora_r  --lora-alpha=$lora_alpha \
    --lora-dropout=$lora_dropout  --orignal-csv-path=$orignal_csv_path  \
    --image-dir=$image_dir  --our-result=$our_result

python ./pandagpt/code/evaluation/evaluate.py -g $compare_result -p $our_result