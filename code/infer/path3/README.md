# path3
This is an implementation of the M4 path-3, which only including embedding and generator.

## Install
Please follow the instructions below to install.

```bash
cd MobileFM
conda create --name path3 python=3.8 -y
conda activate path3

pip install -r code/infer/path3/requirements.txt
```

## Inference
### Task-21 Optical character recongnition
Task-21 optical character recongnition use `Rendered SST2` dataset.
```bash
cd MobileFM
python code/infer/path3/t21-rendered.py
```

### Task-22 Image classification
Task-22 image classification use `cifar100` dataset.
```bash
cd MobileFM
python code/infer/path3/t22-cifar100.py
```

### Task-40 Human activity recognition
Task-40 human activity recognition use `MotionSense` dataset.
```bash
cd MobileFM
python code/infer/path3/t40-motion.py
```
