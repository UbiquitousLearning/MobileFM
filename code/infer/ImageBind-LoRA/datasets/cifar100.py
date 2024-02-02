# import torch
import torch.utils.data as data
# torch.multiprocessing.set_start_method('spawn')
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from models.imagebind_model import ModalityType
import data

class CIFAR100Dataset(CIFAR100):
    def __init__(self, train, transform, device='cpu', datadir="./data/cifar100"):
        self.text_list=["apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel",
           "can","castle","caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach","couch","cra","crocodile","cup","dinosaur","dolphin",
           "elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man",
           "maple_tree","motorcycle","mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plain",
           "plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake",
           "spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train","trout","tulip","turtle",
           "wardrobe","whale","willow_tree","wolf","woman","worm"]
        self.device = device
        super().__init__(datadir, train=train, download=True, transform=transform)
    
    def __getitem__(self, index: int):
        images, target = super().__getitem__(index)
        texts = data.load_and_transform_text([self.text_list[target]], self.device)

        return images, ModalityType.VISION, texts, ModalityType.TEXT
