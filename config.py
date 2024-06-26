# Configuration settings for the project
import albumentations as A
import torch 

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



transforms = A.Compose([
    A.Resize(CITYSCAPE_SIZE[0], CITYSCAPE_SIZE[1]),
])

transforms_GTA5 = A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),


])

augmentations = A.Compose([
    A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
    A.ColorJitter(p=0.5),
    A.GaussianBlur(p=0.5),
])

# We used them to compare different augmentations
# augmentations = {
#     'transform1': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.HorizontalFlip(p=0.5),
#         A.ColorJitter(p=0.5),

#     ]),
#     'transform2': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.ColorJitter(p=0.5),
#         A.GaussianBlur(p=0.5),



#     ]),
#     'transform3': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.HorizontalFlip(p=0.5),
#         A.GaussianBlur(p=0.5),
#     ]),
#     'transform4': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.ColorJitter(p=0.5),
#         A.GaussianBlur(p=0.5),
#         A.GaussNoise(p=0.5),

#     ]),

# }

deeplab_pretrained_model_path = './models/deeplab_resnet_pretrained_imagenet.pth'
cityscapes_path = './Cityscapes'
gta5_path = './GTA5'