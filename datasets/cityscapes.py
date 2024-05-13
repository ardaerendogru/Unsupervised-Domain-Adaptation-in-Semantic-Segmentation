from torch.utils.data import Dataset
from typing import Tuple
import torch 
from PIL import Image
import os

class CityScapes(Dataset):
    """
    CityScapes Dataset class for loading and transforming CityScapes dataset images and labels for semantic segmentation tasks.

    Attributes:
        cityscapes_path (str): Path to the Cityscapes dataset directory.
        train_val (str): Subdirectory to use, typically 'train', 'val', or 'test'.
        transform_image (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        transform_label (callable, optional): A function/transform that takes in a PIL label image and returns a transformed version.
        data (list): List of tuples containing paths to image and corresponding label.
    """
    def __init__(self, cityscapes_path:str, train_val:str,transform_image=None, transform_label=None):

        self.cityscapes_path = cityscapes_path
        self.transform_image = transform_image
        self.transform_label = transform_label
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
        if self.transform_image:
            img = self.transform_image(img)
        if self.transform_label:
            label = self.transform_label(label)
        

        return img, label

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

