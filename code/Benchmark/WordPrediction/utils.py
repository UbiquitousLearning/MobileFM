import os
import sys
import json
from tqdm import tqdm
import math
import cv2
import numpy as np

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from my_dataset import MyDataSet
from params import *


def generate_vocab(data_path):
    word_list = []

    f = open(data_path, 'r')
    lines = f.readlines()
    for sentence in lines:
        word_list += sentence.split()
    word_list = list(set(word_list))

    # 生成字典
    word2index_dict = {w: i + 2 for i, w in enumerate(word_list)}
    word2index_dict['<PAD>'] = 0
    word2index_dict['<UNK>'] = 1
    word2index_dict = dict(sorted(word2index_dict.items(), key=lambda x: x[1]))  # 排序

    index2word_dict = {index: word for word, index in word2index_dict.items()}

    # 将单词表写入json
    json_str = json.dumps(word2index_dict, indent=4)
    with open(vocab_path, 'w') as json_file:
        json_file.write(json_str)

    return word2index_dict, index2word_dict


def generate_dataset(data_path, word2index_dict, n_step=5):
    """
    :param data_path: 数据集路径
    :param word2index_dict: word2index字典
    :param n_step: 窗口大小
    :return: 实例化后的数据集
    """
    def word2index(word):
        try:
            return word2index_dict[word]
        except:
            return 1  # <UNK>

    input_list = []
    target_list = []

    f = open(data_path, 'r')
    lines = f.readlines()
    for sentence in lines:
        word_list = sentence.split()
        if len(word_list) < n_step + 1:  # 句子中单词不足，padding
            word_list = ['<PAD>'] * (n_step + 1 - len(word_list)) + word_list
        index_list = [word2index(word) for word in word_list]
        for i in range(len(word_list) - n_step):
            input = index_list[i: i + n_step]
            target = index_list[i + n_step]

            input_list.append(torch.tensor(input))
            target_list.append(torch.tensor(target))

    # 实例化数据集
    dataset = MyDataSet(input_list, target_list)

    return dataset


def train_one_epoch(model, loss_function, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    # RNNLM based on Attention
    # attention_epoch_path = os.path.join(attention_path, f'{str(epoch).zfill(3)}epoch')
    # os.makedirs(attention_epoch_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, target = data

        pred = model(input.to(device))

        # RNNLM based on Attention
        # pred, attention = model(input.to(device))
        # img = attention.cpu().numpy() * 255
        # img = img.astype(np.uint8)
        # im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        # cv2.imwrite(os.path.join(attention_epoch_path, f'{step}.png'), im_color)

        loss = loss_function(pred, target.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, ppl: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            math.exp(accu_loss.item() / (step + 1)),
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # gradient clip
        # clip_grad_norm_(parameters=model.parameters(), max_norm=0.1, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()
        # update lr
        if lr_scheduler != None:
            lr_scheduler.step()

    return accu_loss.item() / (step + 1), math.exp(accu_loss.item() / (step + 1))

def evaluate_acc(model , data_loader, device, k):
    model.eval()

    all_num = 0
    right_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, target = data
        # print(input.shape)
        pred = model(input.to(device))
        _,indexes = torch.topk(pred,k,dim=1)
        indexes = indexes.cpu().numpy()
        target = target.numpy()
        
        all_num += target.shape[0]
        for i in range(k):
            right_num += (indexes[:,i] == target).sum()
        #print(target)
        #print(indexes[:,0])
        #print(torch.topk(pred,3,dim=1))
        # RNNLM based on Attention
        # pred, _ = model(input.to(device))
    print(all_num,right_num)

    return right_num/all_num



def evaluate(model, loss_function, data_loader, device, epoch):
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, target = data

        pred = model(input.to(device))

        # RNNLM based on Attention
        # pred, _ = model(input.to(device))

        loss = loss_function(pred, target.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, ppl: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            math.exp(accu_loss.item() / (step + 1)),
        )

    return accu_loss.item() / (step + 1), math.exp(accu_loss.item() / (step + 1))


# 调度器，Poly策略
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
