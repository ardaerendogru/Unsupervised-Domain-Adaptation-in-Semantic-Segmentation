# Configuration settings for the project
import albumentations as A
# Batch size for dataloaders
BATCH_SIZE = 4
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


# augmentations = {
#     'transform1': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#     ]),
#     'transform2': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.HueSaturationValue(p=0.5),
#         A.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1), p=0.5),
#     ]),
#     'transform3': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.HorizontalFlip(p=0.5),
#         A.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1), p=0.5),
#         A.GaussNoise(p=0.5),
#     ]),
#     'transform4': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.HorizontalFlip(p=0.5),
#         A.HueSaturationValue(p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#     ]),
#     'transform5': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1), p=0.5),
#         A.GaussNoise(p=0.5),
#     ]),
#     'transform6': A.Compose([
#         A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
#         A.HorizontalFlip(p=0.5),
#         A.GaussNoise(p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.HueSaturationValue(p=0.5),
#     ])
# }


augmentations = {
    'transform1': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.5),

    ]),
    'transform2': A.Compose([
        A.ColorJitter(p=0.5),
        A.GaussianBlur(p=0.5),
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),



    ]),
    'transform3': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(p=0.5),
    ]),
    'transform4': A.Compose([
        A.Resize(GTA5_SIZE[0], GTA5_SIZE[1]),
        A.ColorJitter(p=0.5),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),

    ]),

}

occurence = {'0': 0.4736585289252789,
 '1': 0.037353923869027014,
 '2': 0.1389279692559401,
 '4': 0.007365253267702513,
 '5': 0.01239771468073813,
 '6': 0.0013794206725236096,
 '7': 0.001262954484359789,
 '8': 0.07425178751055177,
 '9': 0.033243068673944144,
 '10': 0.17447357014382447,
 '11': 0.0013711530365722823,
 '12': 0.0002874583073654686,
 '13': 0.02390354978900514,
 '14': 0.01303180359961868,
 '15': 0.002847960219900922,
 '16': 0.003787484910037849,
 '17': 0.00039483666786594736,
 '18': 6.156198574335419e-05}