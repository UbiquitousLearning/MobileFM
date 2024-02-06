# Benchmark
A comprehensive edge-oriented benchmark for AI tasks.

## Task-16 Image retrieval
Task-16 Image retrieval inference on Jetson ORIN.
The implementation is based on [ArcFace](https://github.com/open-mmlab/mmpretrain/tree/17a886cb5825cd8c26df4e65f7112d404b99fe12/configs/arcface) which implemented by [mmpretrain](https://github.com/open-mmlab/mmpretrain/tree/17a886cb5825cd8c26df4e65f7112d404b99fe12).

### Install
Below are quick steps for installation:

```shell
git clone --recurse-submodules -j8 https://github.com/UbiquitousLearning/MobileFM.git
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
cd MobileFM/code/Benchmark/mmpretrain
mim install -e .
```

Please refer to [installation documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for more detailed installation and dataset preparation.

### Inference
Task-16 Image retrieval use `Inshop` dataset.
```shell
cd MobileFM/code/Benchmark/mmpretrain
python tools/test.py configs/arcface/resnet50-arcface_8xb32_inshop.py https://download.openmmlab.com/mmclassification/v0/arcface/resnet50-arcface_inshop_20230202-b766fe7f.pth
```

### Citation
```bibtex
@misc{2023mmpretrain,
    title={OpenMMLab's Pre-training Toolbox and Benchmark},
    author={MMPreTrain Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpretrain}},
    year={2023}
}
```

```bibtex
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```


## Task-1 Input word prediction

### Install
Install the Pytorch package with the correct cuda version, for example

```bash
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch/
```

### Inference
The dataset is PTB dataset, and we already put it in [./penn] directory.
```shell
python test.py
```

### References
[Language-Model-Next-Word-Prediction](https://github.com/friedrichor/Language-Model-Next-Word-Prediction)