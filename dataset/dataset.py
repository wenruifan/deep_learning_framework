from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_list, self.label_list = self._load_data()

    def _load_data(self):
        image_list = [...]
        label_list = [...]

        return image_list, label_list
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = self.label_list[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    


    