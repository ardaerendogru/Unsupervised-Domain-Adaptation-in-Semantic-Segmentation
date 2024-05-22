from torch.utils.data import Dataset
from typing import Tuple
import torch 
from PIL import Image
import os
import numpy as np

class CityScapes(Dataset):
    """
    CityScapes Dataset class for loading and transforming CityScapes dataset images and labels for semantic segmentation tasks.

    Attributes:
        cityscapes_path (str): Path to the CityScapes dataset directory.
        transform (callable, optional): A function/transform that takes in a numpy image and label and returns transformed versions. Expected to be an Albumentations augmentation.
        data (list): List of tuples containing paths to image and corresponding label.
    """
    def __init__(self, cityscapes_path: str, train_val: str, transform=None):
        self.cityscapes_path = cityscapes_path
        self.transform = transform
        self.data = self._load_data(train_val)

    def _load_data(self, train_val: str):
        data = []
        path = os.path.join(self.cityscapes_path, 'Cityspaces', 'gtFine', train_val)
        for root, _, files in os.walk(path):
            for file in files:
                if 'Id' in file:
                    label_path = os.path.join(root, file)
                    img_path = label_path.replace('gtFine/', 'images/').replace('_gtFine_labelTrainIds', '_leftImg8bit')
                    data.append((label_path, img_path))
        return data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label_path, img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        img, label = np.array(img), np.array(label)

        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img, label = transformed['image'], transformed['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()
        return img, label

    def __len__(self) -> int:
        return len(self.data)

