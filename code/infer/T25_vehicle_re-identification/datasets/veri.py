import os
import random
from typing import Callable, Optional

from torch.utils.data import Dataset
import data
import pandas as pd

def load_data(root_dir,listname):
    # load data
    df=pd.DataFrame(index=[i for i in range(777)],columns=['camera','path'])
    with open(os.path.join(root_dir,listname),'r') as f:
        for line in f:
            label = line.split('_')
            if pd.isnull(df.iloc[int(label[0])]).any():
                df.iloc[int(label[0])]=[[],[]]
            df.iloc[int(label[0]),0].append(label[1])
            df.iloc[int(label[0]), 1].append(line[:-1])
    df=df.dropna()
    return df
def get_val_path_list(root_dir,listname):
    paths=[]
    df = load_data(root_dir,listname)
    df = df.reset_index(drop=True)
    for i in range(3):
        for index in df.index:
            ri = random.randint(0, len(df.loc[index, 'camera']) - 1)
            paths.append((df.loc[index, 'path'][ri], index, df.loc[index, 'camera'][ri]))
    return paths
def get_train_path_list(root_dir,listname):
    r'''

        :param root_dir:
        :param listname:
        :return: (path, index in train_embedding,camera_id) index[0,574]
        '''
    paths = []
    df = load_data(root_dir, listname)
    df = df.reset_index(drop=True)
    for index in df.index:
        for i in range(len(df.loc[index, 'camera'])):
            paths.append((df.loc[index, 'path'][i], index, df.loc[index, 'camera'][i]))
    return paths
def get_path_list(root_dir,listname):
    r'''

    :param root_dir:
    :param listname:
    :return: (path,vehicle_id,camera_id)
    '''
    paths=[]
    with open(os.path.join(root_dir, listname), 'r') as f:
        for line in f:
            label = line.split('_')
            paths.append((line[:-1],int(label[0]),label[1]))
    return paths

def gen_train_embedding(root_dir):
    df = load_data(root_dir,'name_train.txt')
    paths=[]
    for index in df.index:
        i = random.randint(0,len(df.loc[index,'camera'])-1)
        paths.append((df.loc[index,'path'][i],index,df.loc[index,'camera'][i]))
    return paths

class VERI(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = 'train', device: str = 'cpu'):
        self.transform = transform
        self.device = device

        if split == 'train':
            self.root_dir = os.path.join(root_dir,'image_train')
            self.paths = get_train_path_list(root_dir,'name_train.txt')
        elif split == 'val':
            self.root_dir = os.path.join(root_dir,'image_train')
            self.paths = get_val_path_list(root_dir,'name_train.txt')
        elif split == 'query':
            self.root_dir = os.path.join(root_dir,'image_query')
            self.paths = get_path_list(root_dir,'name_query.txt')
            self.testlist = get_path_list(root_dir, 'name_test.txt')
        elif split == 'test':
            self.root_dir = os.path.join(root_dir, 'image_test')
            self.paths = get_path_list(root_dir, 'name_test.txt')
        elif split == 'train_emb':
            self.root_dir = os.path.join(root_dir, 'image_train')
            self.paths = gen_train_embedding(root_dir)
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' , 'test' or 'query', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, vehicle_id, camera_id = self.paths[index]
        images = data.load_and_transform_vision_data([os.path.join(self.root_dir,img_path)], self.device, to_tensor=False)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)
        camera_id = data.load_and_transform_text([camera_id], self.device)
        return images, vehicle_id,camera_id

    def image_to_cameraid(self,index):
        '''
        image_index to camera_id
        :return:
        '''
        return self.testlist[index][2]

    def image_to_veid(self,index):
        '''
        image_index to vehicle_id
        :return:
        '''
        return self.testlist[index][1]


def train_embedding(root_dir,veri_list):
    df = load_data(root_dir, 'name_train.txt')
    df = df.reset_index(drop=True)
    paths = []
    for index in veri_list:
        i = random.randint(0, len(df.loc[int(index), 'camera']) - 1)
        paths.append((df.loc[index, 'path'][i], index, df.loc[index, 'camera'][i]))
    return paths

class Embedding(Dataset):
    def __init__(self, root_dir: str, veri_list, transform: Optional[Callable] = None, device: str = 'cpu'):
        self.veri_list = veri_list
        self.transform = transform
        self.device = device
        self.root_dir = os.path.join(root_dir, 'image_train')
        self.paths = train_embedding(root_dir,veri_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, vehicle_id, camera_id = self.paths[index]
        images = data.load_and_transform_vision_data([os.path.join(self.root_dir,img_path)], self.device, to_tensor=False)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)
        camera_id = data.load_and_transform_text([camera_id], self.device)

        return images, vehicle_id,camera_id

