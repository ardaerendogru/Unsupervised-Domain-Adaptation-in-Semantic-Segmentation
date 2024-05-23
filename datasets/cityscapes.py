from torch.utils.data import Dataset
from typing import Tuple
import torch 
from PIL import Image
import os
import numpy as np
from typing import Optional
from albumentations import Compose

class CityScapes(Dataset):


    """
    CityScapes Dataset class for loading and transforming CityScapes dataset images and labels for semantic segmentation tasks.
    """


    def __init__(self, cityscapes_path: str, train_val: str, transform: Optional[Compose] = None):


        """
        Initializes the CityScapes dataset class.

        This constructor sets up the dataset for use, either for training or validation, based on the provided path and subset selection. It optionally applies a transformation to the data.

        Args:
            cityscapes_path (str): The root directory path where the CityScapes dataset is stored.
            train_val (str): A string that specifies whether to load the 'train' or 'val' subset of the dataset.
            transform (callable, optional): A function/transform that takes in an image and label and returns a transformed version. Defaults to None.

        Raises:
            ValueError: If `train_val` is not 'train' or 'val'.
        """


        if train_val not in ['train', 'val']:
            raise ValueError("train_val must be 'train' or 'val'")
        
        self.cityscapes_path = cityscapes_path
        self.transform = transform
        self.data = self._load_data(train_val)

    def _load_data(self, train_val: str)->list:


        """
        Load data paths for CityScapes dataset images and labels.

        This method walks through the directory structure of the CityScapes dataset, specifically looking for label files in the 'gtFine' folder. It constructs the corresponding image paths by replacing parts of the label paths to point to the images in the 'leftImg8bit' folder. It assumes a specific naming convention used by the CityScapes dataset where the label files contain 'Id' in their names.

        Args:
            train_val (str): A string that specifies whether to load the 'train' or 'val' subset of the dataset.

        Returns:
            list: A list of tuples, each containing the path to a label file and the corresponding image file.
        """


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


        """
        Retrieve an item from the dataset at the specified index.

        This method fetches the image and label paths from the dataset, loads the images and labels, applies transformations if specified, and converts them into tensors suitable for model input. The images are converted to RGB and labels are loaded as is. If transformations are provided, they are applied to both the image and the label. Finally, the images and labels are converted to PyTorch tensors, with the image tensor normalized to have values between 0 and 1.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and label tensors.
        """


        label_path, img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        img, label = np.array(img), np.array(label)

        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img, label = transformed['image'], transformed['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()/255
        label = torch.from_numpy(label).long()
        return img, label

    def __len__(self) -> int:


        """
        Get the length of the dataset.

        This method returns the total number of items in the dataset, which corresponds to the number of image-label pairs loaded.

        Returns:
            int: The total number of items in the dataset.
        """

        
        return len(self.data)

