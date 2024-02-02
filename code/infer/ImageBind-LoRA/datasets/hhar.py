import os
import numpy as np

import torch
import torch.utils.data as data

def hhar(train=True):
    root_path = './.datasets/hhar/PreprocessedData/'
    X = np.load(os.path.join(root_path, 'X_{}.npy').format('train' if train else 'test'))
    Y = np.load(os.path.join(root_path, 'Y_{}.npy').format('train' if train else 'test'))
    X = torch.from_numpy(X).float()
    X = X.permute(0, 2, 1)
    Y = torch.from_numpy(Y-1).long()
    dataset = data.TensorDataset(X, Y)
    return dataset

if __name__ == "__main__":
    val_dataset = hhar(train=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=128, shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True)
    for batch_idx, (x, target) in enumerate(val_loader):
        print(x.shape, target.shape)
        break
