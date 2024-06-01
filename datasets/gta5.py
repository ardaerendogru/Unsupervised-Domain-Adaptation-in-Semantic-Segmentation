from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import torch 
from PIL import Image
import os
import numpy as np
import random
from utils import get_color_to_id
from UDA import FDA_transform
from albumentations import Compose

class GTA5(Dataset):


    """
    GTA5 Dataset class for loading and transforming GTA5 dataset images and labels for semantic segmentation tasks.

    """


    def __init__(self, GTA5_path: str, transform: Optional[Compose] = None, FDA: float = None):


        """
        Initializes the GTA5 dataset class.

        This constructor sets up the dataset for use, optionally applying Frequency Domain Adaptation (FDA) and other transformations to the data.

        Args:
            GTA5_path (str): The root directory path where the GTA5 dataset is stored.
            transform (callable, optional): A function/transform that takes in an image and label and returns a transformed version. Defaults to None.
            FDA (float, optional): The beta value for Frequency Domain Adaptation. If None, FDA is not applied. Defaults to None.
        """


        self.GTA5_path = GTA5_path
        self.transform = transform
        self.FDA = FDA
        self.data = self._load_data()
        self.color_to_id = get_color_to_id()
        self.target_images = self._load_target_images() if FDA else []

    def _load_data(self)->List[Tuple[str, str]]:

        """
        Load data paths for GTA5 dataset images and labels.

        This method walks through the directory structure of the GTA5 dataset, specifically looking for image files in the 'images' folder and corresponding label files in the 'labels' folder. It constructs a list of tuples, each containing the path to an image file and the corresponding label file.

        Returns:
            list: A list of tuples, each containing the path to an image file and the corresponding label file.
        """

        data = []
        image_dir = os.path.join(self.GTA5_path, 'images')
        label_dir = os.path.join(self.GTA5_path, 'labels')
        for image_filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_filename)
            label_path = os.path.join(label_dir, image_filename)
            data.append((image_path, label_path))
        return data

    def _load_target_images(self)->List[Tuple[str, str]]:

        """
        Load target images for Frequency Domain Adaptation.

        This method walks through the directory structure of the Cityscapes dataset, specifically looking for image files in the 'gtFine' folder. It constructs a list of tuples, each containing the path to a label file and the corresponding image file.

        Returns:
            list: A list of tuples, each containing the path to a label file and the corresponding image file.
        """

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

        """
        Get the image and label at the specified index.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and label.
        """

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
        

        img = torch.from_numpy(img).permute(2, 0, 1).float()/255
        label = torch.from_numpy(label).long()
        return img, label

    def __len__(self)->int:

        """
        Get the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """

        return len(self.data)
    
    def _convert_rgb_to_label(self, img:Image.Image)->np.ndarray:

        """
        Convert RGB image to grayscale label.

        Args:
            img (Image.Image): The RGB image to convert to grayscale.

        Returns:
            np.ndarray: The grayscale label image.
        """

        gray_img = Image.new('L', img.size)
        label_pixels = img.load()
        gray_pixels = gray_img.load()
        
        for i in range(img.width):
            for j in range(img.height):
                rgb = label_pixels[i, j]
                gray_pixels[i, j] = self.color_to_id.get(rgb, 255)
        
        return gray_img
