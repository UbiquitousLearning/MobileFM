# -*- coding: utf-8 -*-
import tqdm
from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import os
import ipdb
#import gradio as gr
# import mdtex2html
import sys
sys.path.append('./code')
from model.openllama import OpenLLAMAPEFTModel
import torch
import json
import numpy as np
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagebind-ckpt-path', type=str, default='/data/MobileFM/yx/AE/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')
    parser.add_argument('--vicuna-ckpt-path', type=str, default='/data/MobileFM/yx/AE/pretrained_ckpt/llama-7b')
    parser.add_argument('--delta-ckpt-path', type=str, default='/data/MobileFM/yx/AE/KWS/ckpt/pandagpt_7b_v2_peft/pytorch_model.pt')
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--device',type=str, default='cuda:0')
    parser.add_argument('--max-tgt-len', type=int, default=128)
    parser.add_argument('--lora-r', type=int, default=32)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--orignal-json-path', type=str)
    parser.add_argument('--image-dir', type=str)
    

    opt = parser.parse_args()

    return opt



def predict(
    input,
    image_path,
    max_length,
    top_p,
    temperature,
):
    if image_path is None: 
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]

    # prepare the prompt
    prompt_text = ''
    prompt_text += f'{input}'

    response = model.generate({
        'prompt': prompt_text,
        'image_paths': image_path,
        'audio_paths': [],
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
    print(f'[!] init the 7b model over ...')


    orignal_json_path = orig_args.orignal_json_path
    image_dir = orig_args.image_dir
    QA = json.load(open(orignal_json_path,'r'))

    results = []
    right = 0
    for each in tqdm.tqdm(QA.values()):
        response = predict(input=each['question'],
                           image_path=[os.path.join(image_dir, 'COCO_val2014_'+ str(each['image_id']).zfill(12) + '.jpg')],
                           max_length=256,
                           top_p=0.01,
                           temperature=1.0)

        results.append({'image': each['image_id'], 'answer': response})
        if response == each['answer']:
            right += 1
    
    # print("right =",right)
    print('acc =',right/len(QA.keys()))
