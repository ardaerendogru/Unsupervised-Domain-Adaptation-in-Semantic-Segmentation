from torch.utils.data import Dataset
from typing import Tuple
import torch 
from PIL import Image
import os
import numpy as np
from UDA.FDA import FDA_transform
import random

class GTA5(Dataset):
    """
    GTA5 Dataset class for loading and transforming GTA5 dataset images and labels for semantic segmentation tasks.

    Attributes:
        GTA5_path (str): Path to the GTA5 dataset directory.
        transform (callable, optional): A function/transform that takes in a numpy image and label and returns transformed versions.
        FDA (float, optional): The beta value for Frequency Domain Adaptation (FDA) if FDA is to be applied, otherwise None.
        data (list): List of tuples containing paths to image and corresponding label.
        target_images (list): List of tuples containing paths to target images used for FDA, if applicable.
        color_to_id (dict): Mapping from RGB color values to class IDs for segmentation.
    """
    def __init__(self, GTA5_path: str, transform=None, FDA: float = None):
        self.GTA5_path = GTA5_path
        self.transform = transform
        self.FDA = FDA
        self.data = self._load_data()
        self.color_to_id = self._get_color_to_id_mapping()
        self.target_images = self._load_target_images() if FDA else []

    def _load_data(self):
        data = []
        image_dir = os.path.join(self.GTA5_path, 'images')
        label_dir = os.path.join(self.GTA5_path, 'labels')
        for image_filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_filename)
            label_path = os.path.join(label_dir, image_filename)
            data.append((image_path, label_path))
        return data

    def _load_target_images(self):
        target_images = []
        city_path = self.GTA5_path.replace('GTA5', 'Cityscapes')
        city_image_dir = os.path.join(city_path, 'Cityspaces', 'gtFine', 'train')
        for root, _, files in os.walk(city_image_dir):
            for file in files:
                if 'Id' in file:
                    label_path = os.path.join(root, file)
                    image_path = label_path.replace('gtFine/', 'images/').replace('_gtFine_labelTrainIds', '_leftImg8bit')
                    target_images.append((label_path, image_path))
        return target_images

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        label = self._convert_rgb_to_label(Image.open(label_path).convert('RGB'))
        img, label = np.array(img), np.array(label)

        if self.FDA:
            target_image_path = random.choice(self.target_images)[1]
            target_image = Image.open(target_image_path).convert('RGB').resize(img.shape[1::-1])
            img = FDA_transform(img, np.array(target_image), beta=self.FDA)

        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img, label = transformed['image'], transformed['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()
        return img, label

    def __len__(self):
        return len(self.data)
    
    def _convert_rgb_to_label(self, img):
        gray_img = Image.new('L', img.size)
        label_pixels = img.load()
        gray_pixels = gray_img.load()
        
        for i in range(img.width):
            for j in range(img.height):
                rgb = label_pixels[i, j]
                gray_pixels[i, j] = self.color_to_id.get(rgb, 255)
        
        return gray_img

    def _get_color_to_id_mapping(self):
        return {
            (128, 64, 128): 0,   # road
            (244, 35, 232): 1,   # sidewalk
            (70, 70, 70): 2,     # building
            (102, 102, 156): 3,  # wall
            (190, 153, 153): 4,  # fence
            (153, 153, 153): 5,  # pole
            (250, 170, 30): 6,   # light
            (220, 220, 0): 7,    # sign
            (107, 142, 35): 8,   # vegetation
            (152, 251, 152): 9,  # terrain
            (70, 130, 180): 10,  # sky
            (220, 20, 60): 11,   # person
            (255, 0, 0): 12,     # rider
            (0, 0, 142): 13,     # car
            (0, 0, 70): 14,      # truck
            (0, 60, 100): 15,    # bus
            (0, 80, 100): 16,    # train
            (0, 0, 230): 17,     # motorcycle
            (119, 11, 32): 18    # bicycle
        }

