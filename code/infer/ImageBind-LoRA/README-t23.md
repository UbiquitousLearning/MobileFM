# Task-23 Image classification

This is an implementation of the M4 task-23 image classification.
This task using `imagenet` dataset.

Please follow the instructions below to install.
```bash
cd MobileFM
conda create --name t23 python=3.8 -y
conda activate t23

pip install -r code/infer/ImageBind-LoRA/requirements.txt
```

## Inference
```bash
cd MobileFM
python code/infer/ImageBind-LoRA/t23-imagenet.py
```
