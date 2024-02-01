import tqdm
import sys
import torch
import json
import numpy as np
import csv
import argparse
from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import os
import ipdb
#import gradio as gr
# import mdtex2html
sys.path.append('./code')
from model.openllama_whisper import OpenLLAMAPEFTModel




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagebind-ckpt-path', type=str, default='/data/MobileFM/yx/AE/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')
    parser.add_argument('--whisper-ckpt-path', type=str, default='/data/MobileFM/yx/AE/pretrained_ckpt/whisper_ckpt/tiny-en')
    parser.add_argument('--vicuna-ckpt-path', type=str, default='/data/MobileFM/yx/AE/pretrained_ckpt/llama-7b')
    parser.add_argument('--delta-ckpt-path', type=str, default='/data/MobileFM/yx/AE/KWS/ckpt/pandagpt_7b_v2_peft/pytorch_model.pt')
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--device',type=str, default='cuda:0')
    parser.add_argument('--max-tgt-len', type=int, default=128)
    parser.add_argument('--lora-r', type=int, default=32)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--orignal-csv-path', type=str)
    parser.add_argument('--image-dir', type=str)
    

    opt = parser.parse_args()

    return opt



def predict(
    input,
    audio_path,
    max_length,
    top_p,
    temperature,
):
    if audio_path is None:
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]

    # prepare the prompt
    prompt_text = ''
    prompt_text += f'{input}'

    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [],
        'audio_paths': audio_path,
        'video_paths': [],
        'thermal_paths': [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': []
    })
    return response


if __name__ == '__main__':
    orig_args = parse_opt()

    args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': orig_args.imagebind_ckpt_path,
    'whisper_ckpt_path': orig_args.whisper_ckpt_path,
    'vicuna_ckpt_path': orig_args.vicuna_ckpt_path,
    'delta_ckpt_path': orig_args.delta_ckpt_path,
    'stage': orig_args.stage,
    'max_tgt_len': orig_args.max_tgt_len,
    'lora_r': orig_args.lora_r,
    'lora_alpha': orig_args.lora_alpha,
    'lora_dropout': orig_args.lora_dropout,
    }
    model = OpenLLAMAPEFTModel(**args)
    delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().to(torch.device(orig_args.device))
    model.device = torch.device(orig_args.device)
    print(f'[!] init the 13b model over ...')


    # base_path='/data/MobileFM/yx/PandaGPT-inference/WhisperSLU/datasets'
    base_path=orig_args.image_dir
    csv_reader = csv.reader(open(orig_args.orignal_csv_path,'r'))
    # csv_reader = csv.reader(open("/data/MobileFM/yx/PandaGPT-inference/WhisperSLU/datasets/valid-clean.csv"))
    title_flag = True
    results = []
    for row in tqdm.tqdm(csv_reader):
        if title_flag:
            title_flag = False
            continue
        else:
            audio_name = os.path.join(base_path,row[2])
            response = predict(input='What does this person say?',
                           audio_path=[audio_name],
                           max_length=256,
                           top_p=0.01,
                           temperature=1.0)
        results.append({'audio': os.path.basename(audio_name), 'text': response, 'target': row[-2]})
    
    from speechbrain.utils.metric_stats import ErrorRateStats


    wer_metric = ErrorRateStats()

    num = 0
    id = 0
    for each in results:
        wer_metric.append([id], [each['text'].split(' ')], [each['target'].split(' ')])
        id += 1

    wer_metric.write_stats(open(os.path.join(orig_args.image_dir,'./wer.log'),'a+'))
    WER = wer_metric.summarize("error_rate")
    print(f"PandaGPT with Whisper-tiny-en as encoder on FSC-valid, WER: {WER}")
