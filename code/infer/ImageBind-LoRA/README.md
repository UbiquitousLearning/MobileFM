# Path-3
This is an implementation of the M4 path-3, which only including embedding and generator.

## Install
Please follow the instructions below to install.

```bash
cd MobileFM
conda create --name path3 python=3.8 -y
conda activate path3

pip install -r code/infer/ImageBind-LoRA/requirements.txt
```

## Inference
### Task-22 Image classification
Task-22 image classification use `cifar100` dataset.
```bash
cd MobileFM
python code/infer/ImageBind-LoRA/t22-cifar100.py
```

### Task-23 Image classification
Task-23 image classification use `imagenet` dataset.
```bash
cd MobileFM
python code/infer/ImageBind-LoRA/t23-imagenet.py
```
