from typing import Optional, Callable
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from mmengine import list_from_file
from sklearn.model_selection import train_test_split
from PIL import Image

class flickr8k(Dataset):
    def __init__(
        self,
        root_dir: str,
        anne_dir: str,
        transform: Optional[Callable] = None,
        split: str = 'train',
        train_size: float = 0.8, 
        random_seed: int = 43,
    ) -> None:
        self.root_dir = root_dir
        self.anne_dir = anne_dir
        self.transform = transform
        if self.transform == None:
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
        self.lines = list_from_file(self.anne_dir)
        self.paths = []
        random.seed(random_seed)
        for idx in range(0, len(self.lines), 5):
            idx_plus = idx + random.randint(0, 4)
            line = self.lines[idx_plus]
            img, text = line.split('.jpg,')
            text = text.strip('"').strip("'").strip('.').strip(' ').lower()
            img_name = img + ".jpg"
            img_path = os.path.join(self.root_dir, img_name)
            self.paths.append((img_path, text, img))
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
        img_path, label, img_name = self.paths[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_name

class flickr30k(Dataset):
    def __init__(
        self,
        root_dir: str,
        anne_dir: str,
        transform: Optional[Callable] = None,
        split: str = 'train',
        train_size: float = 0.8, 
        random_seed: int = 43,
    ) -> None:
        self.root_dir = root_dir
        self.anne_dir = anne_dir
        self.transform = transform
        if self.transform == None:
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
        self.lines = list_from_file(self.anne_dir)
        self.paths = []
        random.seed(random_seed)
        for idx in range(1, len(self.lines), 5):
            idx_plus = idx + random.randint(0, 4)
            line = self.lines[idx_plus]
            img_name, _, text = line.split('|')
            img_name = img_name.strip(' ')
            img = img_name.strip('.jpg')
            text = text.strip('"').strip("'").strip('.').strip(' ').lower()
            img_path = os.path.join(self.root_dir, img_name)
            self.paths.append((img_path, text, img))
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
        img_path, label, img_name = self.paths[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_name
