import os
import numpy as np

import torch
import torch.utils.data as data

from typing import Tuple

import torch
from torch import Tensor
from torchvision.datasets import Flickr8k

from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torchvision import transforms

class flickr8k(Dataset):
    def __init__(self, root_dir: str, anne_dir : str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.anne_dir = anne_dir
        self.transform = transform
        self.device = device
        
        self.lines = list_from_file(self.anne_dir)
        self.paths = []
        self.image_list = []
        for line in self.lines:
            img, text = line.split('.jpg,')
            img_name = img + ".jpg"
            img_path = os.path.join(self.root_dir, img_name)
            self.paths.append((img_path, text , img))

        # Split dataset
        train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
            
        elif split == 'test':
            self.paths = test_paths
            
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")
        
        # for img_path, label ,img_name in self.paths:
        #     self.image_list.append(img_name)
        
        
        # torch.save(self.image_list,"flickr8k_image_list")
        
            
        
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, label ,img_name = self.paths[index]
        image = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        
        return image , label , img_name

class Flickr30kDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        image_name = self.img_names[idx].split('.')[0]
        return image, image_name

image_dir = '/home/bingxing2/gpuuser590/gsy/ImageBind-LoRA-main/data/flickr30k/flickr30k-images'

dataset = Flickr30kDataset(image_dir)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)