import torch
from torch import nn
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, item):
        input = self.inputs[item]
        target = self.targets[item]

        return input, target

    def __len__(self):
        return len(self.inputs)
