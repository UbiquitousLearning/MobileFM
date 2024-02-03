import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mmengine import list_from_file
from PIL import Image

class Inshop(Dataset):
    def __init__(self, ann_path, image_path, split='train', random_seed=43):
        random.seed(random_seed)
        self.image_label_list = self.list_split(ann_path, image_path, split)
        self.image_path = image_path
        self.split = split
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
    
    def __getitem__(self, index):
        image_path, label = self.image_label_list[index]
        img = Image.open(image_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.image_label_list)
    
    def list_split(self, ann_path, image_path, split):
        if split not in ['train', 'train_gallery', 'query', 'gallery']:
            raise ValueError("argument: 'split' must be one of ['train', 'train_gallery', 'query', 'gallery']")
        lines = list_from_file(ann_path)[2:]
        image_label_list = []
        if split != 'train_gallery':
            for line in lines:
                img_name, item_id, status = line.split()
                if status == split:
                    img_path = os.path.join(image_path, img_name)
                    item_id = int(item_id.split('_')[-1])
                    image_label_list.append((img_path, item_id))
            random.shuffle(image_label_list)
        else: # split == 'train_gallery':
            for line in lines:
                img_name, item_id, status = line.split()
                if status == 'train':
                    img_path = os.path.join(image_path, img_name)
                    item_id = int(item_id.split('_')[-1])
                    if not image_label_list or image_label_list[-1][1] != item_id:
                        image_label_list.append((img_path, item_id))
                    elif random.randint(0, 2):
                        image_label_list[-1] = (img_path, item_id)
            image_label_list.sort(key=lambda x: x[1])
        return image_label_list

if __name__ == '__main__':
    batch_size = 128
    ann_file = "./.datasets/inshop/Eval/list_eval_partition.txt"
    image_dir = "./.datasets/inshop/Img"
    test_ds = Inshop(ann_file, image_dir, split='train_gallery')
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)
    for batch_idx, (x, target) in enumerate(test_dl):
        print(x.shape, target.shape)
        break
