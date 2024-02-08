import json
from datasets import load_dataset
from transformers import pipeline
from evaluate import load
import evaluate
import torch
import time
from tqdm import tqdm   

device='cuda'
# raw_data=load_dataset("jfleg")
# val_set=raw_data['validation']
# test_set=raw_data['test']
f=open(file='jfleg_data_dev.json',mode='r',encoding='utf-8')
data=json.load(f)

model_name="pszemraj/flan-t5-large-grammar-synthesis"
text2text_generationer=pipeline(
    task='text2text-generation',
    model=model_name,
    tokenizer=model_name,
    device=7
)
total_count=0
total_bleu=0
bleu = load("bleu")
start=time.time()
for example in data:
    total_count+=1
    sentence=example['sentence']
    if sentence=='':
        continue
    corrections=[example['correction']]
    now=time.time()
    predictions=[text2text_generationer(sentence)[0]['generated_text']]
    pred_time=time.time()-now
    now=time.time()
    results = bleu.compute(predictions=predictions, references=corrections)
    bleu_time=time.time()-now
    now=time.time()
    total_bleu+=results['bleu']
    print(f"example: {total_count}, now_bleu: {results['bleu']}, average_bleu: {total_bleu/total_count}, pred_time: {pred_time}, bleu_time: {bleu_time}")

print(f"total_count: {total_count}, average_bleu: {total_bleu/total_count}, time_cost: {time.time()-start}")



