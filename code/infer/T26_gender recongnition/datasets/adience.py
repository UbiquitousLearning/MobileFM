import os
from typing import Callable, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import data
import pandas as pd

def load_data(root_dir):
    # load data
    data = pd.read_csv(root_dir+"/fold_0_data.txt",sep="\t")
    data1 = pd.read_csv(root_dir+"/fold_1_data.txt",sep="\t")
    data2 = pd.read_csv(root_dir+"/fold_2_data.txt",sep="\t")
    data3 = pd.read_csv(root_dir+"/fold_3_data.txt",sep="\t")
    data4 = pd.read_csv(root_dir+"/fold_4_data.txt",sep="\t")
    alldata = pd.concat([data,data1,data2,data3,data4],ignore_index=True)
    alldata = alldata[(alldata.gender=='f') | (alldata.gender=='m')]
    #print(alldata.shape)
    return alldata


class Adience(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = os.path.join(root_dir,'aligned/')
        self.transform = transform
        self.device = device

        self.classes = ["male","female"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}


        #todo:遍历图片 将图片路径-标签存入paths
        alldata = load_data(root_dir)
        self.paths = []
        for index in alldata.index:
            path = self.root_dir + str(alldata.loc[index,'user_id'])+'/landmark_aligned_face.'+str(alldata.loc[index,'face_id'])+'.'+ str(alldata.loc[index,'original_image'])
            #self.paths.append((path,'male' if alldata.loc[index,'gender']=='m' else 'female'))
            self.paths.append((path, 0 if alldata.loc[index, 'gender'] == 'm' else 1))
        # Split dataset
        train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
        elif split == 'test':
            self.paths = test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, class_id = self.paths[index]
        images = data.load_and_transform_vision_data([img_path], self.device, to_tensor=False)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        #texts = data.load_and_transform_text([class_text], self.device)

        return images, class_id


#a=Adience("../.datasets/adience")