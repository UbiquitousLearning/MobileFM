import os
import numpy as np

import torch
import torch.utils.data as data

def hhar(train=True):
    root_path = 'code/datasets/hhar/PreprocessedData/'
    X = np.load(os.path.join(root_path, 'X_{}.npy').format('train' if train else 'test'))
    Y = np.load(os.path.join(root_path, 'Y_{}.npy').format('train' if train else 'test'))
    X = torch.from_numpy(X).float()
    X = X.permute(0, 2, 1)
    Y = torch.from_numpy(Y-1).long()
    dataset = data.TensorDataset(X, Y)
    return dataset

def motion(train=True):
    root_path = 'code/datasets/motionsense/A_DeviceMotion_data/PreprocessedData-50hz/'
    X = np.load(os.path.join(root_path, 'X_{}.npy').format('train' if train else 'test'))
    Y = np.load(os.path.join(root_path, 'Y_{}.npy').format('train' if train else 'test'))
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()
    dataset = data.TensorDataset(X, Y)
    return dataset

def mhealth(train=True):
    root_path = 'code/datasets/mhealth/PreprocessedData/'
    X = np.load(os.path.join(root_path, 'X_{}.npy').format('train' if train else 'test'))
    Y = np.load(os.path.join(root_path, 'Y_{}.npy').format('train' if train else 'test'))
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y-1).long()
    dataset = data.TensorDataset(X, Y)
    return dataset
