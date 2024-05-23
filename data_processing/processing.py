import torch
from typing import Tuple

def calculate_mean_std(dataloader:torch.utils.data.DataLoader)->Tuple[torch.Tensor, torch.Tensor]:

    """
    Calculate the mean and standard deviation of the dataset provided by the dataloader.

    This function iterates over the dataloader to compute the mean and standard deviation of the entire dataset.
    It is useful for normalizing the dataset in future processing steps.

    Parameters:
        dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and standard deviation tensors.
    """
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std

