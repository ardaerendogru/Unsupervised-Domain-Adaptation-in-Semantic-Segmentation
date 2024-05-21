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
    def __init__(self, cityscapes_path:str, train_val:str, transform = None):

        self.cityscapes_path = cityscapes_path
        self.transform = transform
        self.data = []
        for root, dirs, files in os.walk(f'{cityscapes_path}/Cityspaces/gtFine/{train_val}'): 
            for file in files:
                if 'Id' in file:
                    self.data.append((f'{root}/{file}',f'{root}/{file}'.replace('gtFine/','images/').replace('_gtFine_labelTrainIds', '_leftImg8bit' )))
        


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        "Returns one sample of data (X, y)."
        label_path, img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')  # Ensure image is RGB
        label = Image.open(label_path)
        img = np.array(img) 
        label = np.array(label)
        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()
        return img, label

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

