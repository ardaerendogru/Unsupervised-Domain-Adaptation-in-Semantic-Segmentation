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
    def __init__(self, GTA5_path:str, transform=None, FDA = None):
        self.GTA5_path = GTA5_path
        self.transform = transform
        self.data = []
        self.target_images = []
        self.FDA = FDA
        for path in os.listdir(f'{GTA5_path}/images'):
            self.data.append((f'{GTA5_path}/images/{path}',f'{GTA5_path}/labels/{path}'))
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
        if self.FDA:
            city_path = GTA5_path.replace('GTA5', 'Cityscapes')
            for root, dirs, files in os.walk(f'{city_path}/Cityspaces/gtFine/train'): 
                for file in files:
                    if 'Id' in file:
                        self.target_images.append((f'{root}/{file}',f'{root}/{file}'.replace('gtFine/','images/').replace('_gtFine_labelTrainIds', '_leftImg8bit' )))
        

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        "Returns one sample of data (X, y)."
        img_path, label_path = self.data[index]
        img = Image.open(img_path).convert('RGB')  # Ensure image is RGB
        label = self.convert_rgb_to_label(Image.open(label_path).convert('RGB'))
        img = np.array(img)
        label = np.array(label)
        if self.FDA:
            target_image_pil = Image.open(random.choice(self.target_images)[1]).convert('RGB').resize((img.shape[1], img.shape[0]))
            target_image = np.array(target_image_pil)
            img = FDA_transform(img, target_image, beta=self.FDA)
        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']

            

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()
        return img, label #, fda_transformed

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

