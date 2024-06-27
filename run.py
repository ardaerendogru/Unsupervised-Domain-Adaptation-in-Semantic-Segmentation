import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from models.bisenet.build_bisenet import BiSeNet
from config import (BATCH_SIZE, NC, 
                    NUM_WORKERS, CITYSCAPE_SIZE, transforms, deeplab_pretrained_model_path)
import argparse
import numpy as np
from utils.utilities import get_id_to_color

class CustomImageDataset(Dataset):
    def __init__(self, images_path, transforms=transforms):
        self.images_path = images_path
        self.image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return torch.tensor(image).permute(2, 0, 1).float() / 255.0, os.path.basename(img_path)

def save_predictions_as_images(predictions, file_names, save_path, height, width):
    id_to_color = get_id_to_color()
    os.makedirs(save_path, exist_ok=True)
    
    for pred, file_name in zip(predictions, file_names):
        pred = pred.numpy()
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for label, color in id_to_color.items():
            color_image[pred == label] = color
        
        img = Image.fromarray(color_image)
        base_name, ext = os.path.splitext(file_name)
        img.save(os.path.join(save_path, f"{base_name}_out{ext}"))

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the dataset
    dataset = CustomImageDataset(args.images_path, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load the model
    if args.model == 'bisenet18':
        model = BiSeNet(NC, 'resnet18').to(device)
    elif args.model == 'bisenet101':
        model = BiSeNet(NC, 'resnet101').to(device)
    elif args.model == 'deeplabv2':
        model = get_deeplab_v2(num_classes=19, pretrain=False).to(device)
    else:
        raise ValueError("Unsupported model")

    # Load the checkpoint
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    # Evaluate the model
    results = []
    file_names = []
    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            results.extend(preds.cpu())
            file_names.extend(names)

    # Save the results as images
    save_predictions_as_images(results, file_names, args.save_path, height=CITYSCAPE_SIZE[0], width=CITYSCAPE_SIZE[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bisenet18', help='Model to use')
    parser.add_argument('--images_path', type=str, required=True, help='Path to the images')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.pth', help='Path to the checkpoint')
    parser.add_argument('--save_path', type=str, default='./results/labels', help='Path to save the labels')
    args = parser.parse_args()
    main(args)
