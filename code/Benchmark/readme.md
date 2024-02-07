# eAIBench: a comprehensive edge-oriented benchmark for AI tasks.
We embark on constructing an exhaustive edge-oriented benchmark for AI tasks, encompassing 38 tasks spanning 50 public datasets.
Those tasks are essential to real-world mobile applications (e.g., translation, object detection, and voice assistant). Each task is accompanied by its designated accuracy metric. eAIBench includes 5 modality domains: NLP, CV, Audio, Sensing (IMU), and Misc (Multimodal). While the majority of tasks are tailored to smartphones, we extend our scope to encompass pivotal devices such as laptops (code generation), autonomous cars (traffic sign classification), and IoT cameras (Counting).

# Task deployment
Given the dispersed nature of the current 50 tasks, we will initially prioritize displaying the code and dataset of select tasks in the first stage. Subsequently, we will devise a more convenient approach to release all tasks comprehensively.

## Task-16 Image retrieval
Task-16 Image retrieval inference on Jetson ORIN.
The implementation is based on [ArcFace](https://github.com/open-mmlab/mmpretrain/tree/17a886cb5825cd8c26df4e65f7112d404b99fe12/configs/arcface) which implemented by [mmpretrain](https://github.com/open-mmlab/mmpretrain/tree/17a886cb5825cd8c26df4e65f7112d404b99fe12).

### Install
Below are quick steps for installation:

```shell
git clone --recurse-submodules -j8 https://github.com/UbiquitousLearning/MobileFM.git
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install -U openmim
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

## Task-19 Semantic segmentation
Task-19 Semantic segmentation inference on Jetson ORIN.
The implementation is based on [Deeplabv3+](https://github.com/open-mmlab/mmsegmentation/tree/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8/configs/deeplabv3plus) which implemented by [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8).

### Install
Below are quick steps for installation:
```shell
git clone --recurse-submodules -j8 https://github.com/UbiquitousLearning/MobileFM.git
conda create -n open-mmlab python=3.8
conda activate open-mmlab
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
cd MobileFM/code/Benchmark/mmsegmentation
pip install -v -e .
```

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/tree/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8/docs/en/get_started.md#installation) for installation and [dataset_prepare.md](https://github.com/open-mmlab/mmsegmentation/tree/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

### Inference
Task-19 Semantic segmentation use `ADE20K` dataset.
```shell
cd MobileFM/code/Benchmark/mmsegmentation
python tools/test.py configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth
```

## Task-20 Semantic segmentation
Task-20 Semantic segmentation inference on Jetson ORIN.
The implementation is based on [Deeplabv3+](https://github.com/open-mmlab/mmsegmentation/tree/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8/configs/deeplabv3plus) which implemented by [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8).

### Install
Please refer to [Task-19 install](https://github.com/UbiquitousLearning/MobileFM/blob/main/code/Benchmark/readme.md) for installation.

### Inference
Task-20 Semantic segmentation use `Pascal VOC 2012` dataset.
```shell
cd MobileFM/code/Benchmark/mmsegmentation
python tools/test.py configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-40k_voc12aug-512x512.py https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth
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
