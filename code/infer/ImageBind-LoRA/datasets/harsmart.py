import os
import numpy as np

import torch
import torch.utils.data as data

def HARsmart_origin(train=True):
    root_path = './.datasets/harsmart_origin/PreprocessedData/'
    X = np.load(os.path.join(root_path, 'X_{}.npy').format('train' if train else 'test'))
    Y = np.load(os.path.join(root_path, 'Y_{}.npy').format('train' if train else 'test'))
    X = torch.from_numpy(X).float()
    X = X.permute(0, 2, 1)
    Y = torch.from_numpy(Y-1).long()
    dataset = data.TensorDataset(X, Y)
    return dataset

def HARsmart(train=True):
    input_names = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    root_path = './.datasets/UCI HAR Dataset/'
    if train:
        root_path = os.path.join(root_path, 'train')
    else:
        root_path = os.path.join(root_path, 'test')
    root_path_x = os.path.join(root_path, 'Inertial Signals')
    X = []
    for input_name in input_names:
        X_temp = torch.from_numpy(np.loadtxt(os.path.join(root_path_x, input_name + '_' + ('train' if train else 'test') + '.txt'))).float()
        X_temp = X_temp.repeat(1, 16)[:, :2000]
        X.append(X_temp)
    X = torch.stack(X, dim=1)
    Y = torch.from_numpy(np.loadtxt(os.path.join(root_path, 'y_' + ('train' if train else 'test') + '.txt'))-1).long()
    dataset = data.TensorDataset(X, Y)
    return dataset

if __name__ == "__main__":
    val_dataset = HARsmart_origin(train=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=128, shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True)
    for batch_idx, (x, target) in enumerate(val_loader):
        print(x.shape, target.shape)
        break
