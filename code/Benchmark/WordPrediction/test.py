import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import generate_dataset, evaluate, evaluate_acc
from params import *


def main():
    print(f"using {device} device.")
    print("Using {} dataloader workers every process".format(nw))

    # 读取单词表
    with open(vocab_path, "r") as f:
        word2index_dict = json.load(f)
    index2word_dict = {index: word for word, index in word2index_dict.items()}

    n_class = len(word2index_dict)  # number of Vocabulary
    print("number of Vocabulary =", n_class)

    # 实例化数据集
    test_dataset = generate_dataset(test_path, word2index_dict, n_step)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
    )

    model = torch.load(test_model).to(device)

    # test
    # 这里测试步骤和验证一样，所以直接使用evaluate函数了
    for k in range(1,6,2):
        acc = evaluate_acc(model=model,data_loader=test_loader,device=device,k=k)
        print(f'Top-{k} accuracy = {acc}')




if __name__ == "__main__":
    main()
