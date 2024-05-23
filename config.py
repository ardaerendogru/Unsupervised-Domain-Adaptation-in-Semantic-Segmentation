# Configuration settings for the project
import albumentations as A
# Batch size for dataloaders
BATCH_SIZE = 2
EPOCHS = 50
# Number of classes
NC = 19

# Number of workers for DataLoader
NUM_WORKERS = 8

# Image sizes for different datasets
CITYSCAPE_SIZE = (512, 1024)
GTA5_SIZE = (720, 1280)

# Mean and standard deviation for normalization - Cityscapes
CITYSCAPES_MEAN = [78.5451, 87.7702, 76.9778]
CITYSCAPES_STD = [47.7790, 48.5031, 47.8005]

# Mean and standard deviation for normalization - GTA5
GTA5_MEAN = [129.5363, 127.9398, 123.2765]
GTA5_STD = [63.8022, 62.4980, 62.0512]


augmentations = {
    'transform1': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ]),
    'transform2': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1), p=0.5),
    ]),
    'transform3': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1), p=0.5),
        A.GaussNoise(p=0.5),
    ]),
    'transform4': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ]),
    'transform5': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1), p=0.5),
        A.GaussNoise(p=0.5),
    ]),
    'transform6': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
    ])
}