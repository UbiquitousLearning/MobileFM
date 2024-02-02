## Environment Installation:

To install the required environment, please run

```bash
pip install -r requirements.txt
```

Then install the Pytorch package with the correct cuda version, for example

```bash
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch/
```

## Prepare ImageBind Checkpoint:
You can download the pre-trained ImageBind model using https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth. After downloading, put the downloaded file (imagebind_huge.pth) in [./pandagpt/ckpt/imagebind] directory.

## Prepare Vicuna Checkpoint:
To prepare the pre-trained Vicuna model, please follow the instructions provided https://github.com/yxuansu/PandaGPT/blob/main/pretrained_ckpt#1-prepare-vicuna-checkpoint. After downloading, put the downloaded file in [./pandagpt/ckpt/llama-7b/] directory.

## Prepare Whisper Checkpoint：

You can download the pre-trained Whisper model using https://huggingface.co/openai/whisper-tiny.en. After downloading, put the downloaded file in [./pandagpt/ckptl/whisper_ckpt/tiny-en] directory.

## Inference

### Visual question answering 

Download dataset of VQA v2.0(only val dataset part) and put the datasets in [./pandagpt/data/VQA/images] directory. 
```bash
./VQA.sh
```

### Spoken language understanding
Download dataset of FSC and put the datasets in [./pandagpt/data/FSC/speakers] directory. 
```bash
./FSC.sh
```

Download dataset of SLURP and put the datasets in [./pandagpt/data/SLURP/slurp_real] directory. 
```bash
./SLURP.sh
```

### Keyword Spotting 
Download dataset of Speech command  and put the datasets in [./pandagpt/data/KWS/dataset] directory. 
```bash
./KWS.sh
```
