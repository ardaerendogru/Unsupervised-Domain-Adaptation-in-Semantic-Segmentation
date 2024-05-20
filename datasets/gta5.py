from torch.utils.data import Dataset
from typing import Tuple
import torch 
from PIL import Image
import os
import numpy as np


class GTA5(Dataset):
    """
    GTA5 Dataset class for loading and transforming GTA5 dataset images and labels for semantic segmentation tasks.

    Attributes:
        GTA5_path (str): Path to the GTA5 dataset directory.
        transform_image (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        transform_label (callable, optional): A function/transform that takes in a PIL label image and returns a transformed version.
        augmentations (callable, optional): A function/transform that takes in both the image and label and returns transformed versions.
        data (list): List of tuples containing paths to image and corresponding label.
        color_to_id (dict): Mapping from RGB color values to class IDs.
    """
    def __init__(self, GTA5_path:str,transform_image=None, transform_label=None, augmentations=None):
        self.GTA5_path = GTA5_path
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.augmentations = augmentations
        self.data = []
        self.color_to_id = {
            (128, 64, 128):0,   #road
            (244, 35, 232): 1,  # sidewalk
            (70, 70, 70): 2,    # building
            (102, 102, 156): 3, # wall
            (190, 153, 153): 4, # fence
            (153, 153, 153): 5, # pole
            (250, 170, 30): 6,  # light
            (220, 220, 0): 7,   # sign
            (107, 142, 35): 8,  # vegetation
            (152, 251, 152): 9, # terrain
            (70, 130, 180): 10, # sky
            (220, 20, 60): 11,  # person
            (255, 0, 0): 12,    # rider
            (0, 0, 142): 13,    # car
            (0, 0, 70): 14,     # truck
            (0, 60, 100): 15,   # bus
            (0, 80, 100): 16,   # train
            (0, 0, 230): 17,    # motorcycle
            (119, 11, 32): 18   # bicycle
        }

        for path in os.listdir(f'{GTA5_path}/images'):
            self.data.append((f'{GTA5_path}/images/{path}',f'{GTA5_path}/labels/{path}'))
            
        


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        "Returns one sample of data (X, y)."
        img_path, label_path = self.data[index]
        img = Image.open(img_path).convert('RGB')  # Ensure image is RGB
        label = self.convert_rgb_to_label(Image.open(label_path).convert('RGB'))
        if self.transform_image and not self.augmentations:
            img = self.transform_image(img)
        if self.transform_label and not self.augmentations:
            label = self.transform_label(label)
        
        if self.augmentations:
            img = np.array(img)
            label = np.array(label)
            transformed = self.augmentations(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            label = torch.from_numpy(label).long()

        return img, label

    def __len__(self):
        return len(self.data)
    
    def convert_rgb_to_label(self, img):
        """
        Converts an RGB image to a label image where each pixel's value corresponds to a class ID.

        Args:
            img (PIL.Image): The RGB image to be converted.

        Returns:
            PIL.Image: A grayscale image where each pixel's intensity represents a class ID.
        """
        # Convert label image to RGB if not already
        label_img = img
        # Create a new grayscale image with the same size
        gray_img = Image.new('L', label_img.size)
        # Load pixel data
        label_pixels = label_img.load()
        gray_pixels = gray_img.load()
        
        # Map each pixel to grayscale
        for i in range(label_img.width):
            for j in range(label_img.height):
                rgb = label_pixels[i, j]
                if rgb in self.color_to_id.keys():
                    gray_pixels[i, j] = self.color_to_id[rgb]
                else:
                    gray_pixels[i, j] = 255
        
        return gray_img

