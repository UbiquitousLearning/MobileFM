import math
import json
import sys
import time
from pathlib import Path
from typing import Optional
import logging

import lightning as L
import torch
import tqdm
import argparse

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from lit_llama.lora import lora
from generate import generate

def generate_prompt_tc(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    return f"### Sentence:\n{example['sentence']}\n\n### Category:\n"


lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
max_seq_length = 512


def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0

  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)

  return 2 * (prec * rec) / (prec + rec)


def model_predict(sentence, model, tokenizer):

    full_prompt = generate_prompt_tc({"sentence":sentence})
    encoded = tokenizer.encode(full_prompt, bos=True, eos=False, device=model.device)
    prompt_input_len = len(encoded)

    output, probs = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
        temperature=1,
        eos_id=tokenizer.eos_id,
        argmax=True
    )
    return tokenizer.decode(output[prompt_input_len:-1]), probs


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(logging.StreamHandler()) # print to terminal also (logging.info)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def main(
    squad_dataset_path: Optional[Path] = None,
    *,
    accelerator: str = "cuda",
    lora_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    dtype: str = "bfloat16",
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer
       finetuned with LoRA.

    Args:
        datasets: The datasets to use as a comma separated string
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    if not lora_path:
        lora_path = Path("./lit-llama/checkpoints/lora/agnews/lit-sst-llama-lora-8-finetuned-agnews.pth")
    if not checkpoint_path:
        checkpoint_path = Path(f"./model/lit-llama/7B/lit-llama.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./model/lit-llama/tokenizer.model")
    if not squad_dataset_path:
        squad_dataset_path = Path("./lit-llama/data/agnews/agnews_data_dev.json")

    assert lora_path.is_file()
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()
    assert squad_dataset_path.is_file()

    with open(squad_dataset_path, "rb") as f:
        squad_eval_data = json.load(f)


    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")


    fabric = L.Fabric(accelerator=accelerator, devices=[7])
    fabric.seed_everything(42)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Loading model ...", file=sys.stderr)
    log_name = "".join(str(lora_path).split("/")[-1].split(".")[:-1]) + "-only-argmax"
    log_path =  lora_path.parent
    t0 = time.time()
    logger = setup_logger("Squad Evaluation Log", Path(log_path / f"squad_evaluation_{log_name}.log"))

    with lazy_load(checkpoint_path) as pretrained_checkpoint, lazy_load(lora_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned adapter weights
            model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    # if compile:
    #     model = torch.compile(model)
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)

    t0 = time.time() # reset

    predict_dict = {}
    prob_dict = {}
    total_count= 0
    correct_count=0
    example_count=len(squad_eval_data)
    unk_count = 0
    unk_gt_count = 0
    f1_count = 0
    em_score = 0
    f1_score = 0
    # Search for each passage, its question and its answer
    for example in squad_eval_data:
        sentence=example['sentence']
        true_answer=example['category']
        model_answer, model_prob = model_predict(sentence, model, tokenizer)
        total_count+=1
        if model_answer==true_answer:
            correct_count+=1
        print(f"total_count: {example_count}, now_count: {total_count}, now_ACC: {correct_count/total_count}, "
              f"model answer: {model_answer}, true answer: {true_answer}, model prob: {model_prob}")


    logger.info(f"Total ACC: {correct_count/total_count}")
    logger.info(f"Inference through whole dataset: {time.time() - t0:.02f} seconds.\n")

    # # Save predictions for further evaluation by official script (EM and F1)
    # with open(Path(log_path / f"squad_eval_predict_{log_name}.json"), "w") as fp:
    #     json.dump(predict_dict, fp)
    # with open(Path(log_path / f"squad_eval_probs_{log_name}.json"), "w") as fp:
    #     json.dump(prob_dict, fp)

    logger.info(
        f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
