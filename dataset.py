import torch
import torchvision
import os
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from torchvision.io import read_image, ImageReadMode 
from PIL import Image

#custom dataset for didatask
class SatelliteDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.transform = transform

        self.images = []
        self.masks = []

        for image in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, image))
        
        for mask in os.listdir(mask_dir):
            self.masks.append(os.path.join(mask_dir, mask))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        
        if self.transform:
            image, mask = self.transform(image, mask)
            mask = mask.unsqueeze(dim=0)
        
        return image, mask