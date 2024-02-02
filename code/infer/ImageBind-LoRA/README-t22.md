# Task-22 Image classification

This is an implementation of the M4 task-22 image classification.
This task using `cifar100` dataset.

Please follow the instructions below to install.
```bash
cd MobileFM
conda create --name t22 python=3.8 -y
conda activate t22

pip install -r code/infer/ImageBind-LoRA/requirements.txt
```

## Inference
```bash
cd MobileFM
python code/infer/ImageBind-LoRA/imagebind-none-cifar100.py
```
