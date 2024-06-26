import torch
from torch.utils.data import DataLoader
import albumentations as A
from datasets import GTA5, CityScapes
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from models.bisenet.build_bisenet import BiSeNet
from training import train
from model_evaluation import save_results
from visualization import plot_loss, plot_mIoU, plot_IoU
from config import (BATCH_SIZE, NC, 
                    NUM_WORKERS, CITYSCAPE_SIZE, GTA5_SIZE, EPOCHS,
                    augmentations, transforms, transforms_GTA5, deeplab_pretrained_model_path,
                    cityscapes_path, gta5_path)
import argparse


def main(args):
    
    torch.cuda.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'cityscapes':
        train_dataset = CityScapes(cityscapes_path, 'train', transform=transforms)
        val_dataset = CityScapes(cityscapes_path, 'val', transform=transforms)

    elif args.dataset == 'gta5':
        if args.augmentation:
            if args.fda is not None:
                train_dataset = GTA5(gta5_path, transform=augmentations, FDA=args.fda)
            else:
                train_dataset = GTA5(gta5_path, transform=augmentations)
            val_dataset = CityScapes(cityscapes_path, 'val', transform=transforms)
        else:
            train_dataset = GTA5(gta5_path, transform=transforms)
            val_dataset = CityScapes(cityscapes_path, 'val', transform=transforms)
    else:
        raise ValueError("Unsupported dataset")
    
    if args.model == 'bisenet18':
        model = BiSeNet(NC, 'resnet18').to(device)
    elif args.model == 'bisenet101':
        model = BiSeNet(NC, 'resnet101').to(device)
    elif args.model == 'deeplabv2':
        model = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path=deeplab_pretrained_model_path).to(device)
    else:
        raise ValueError("Unsupported model")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    if args.dacs:
        target_dataset = CityScapes(cityscapes_path, 'train', transform=transforms)
        target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unsupported optimizer")

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    if args.dacs:
        results = train(model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        train_dataloader=train_dataloader,
                        validation_dataloader=val_dataloader,
                        target_dataloader=target_dataloader,
                        epochs=EPOCHS,
                        device=device,
                        model_name=args.model,
                        class_number=NC)
    else:
        results = train(model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        train_dataloader=train_dataloader,
                        validation_dataloader=val_dataloader,
                        epochs=EPOCHS,
                        device=device,
                        model_name=args.model,
                        class_number=NC)
    
    save_results(model, results, f"{args.results_path}{args.save_name}", height=CITYSCAPE_SIZE[0], width=CITYSCAPE_SIZE[1], iterations=100)
    plot_loss(results, f"{args.model}", f"{args.save_name}", f"{args.dataset}", "CityScapes")
    plot_mIoU(results, f"{args.model}", f"{args.save_name}", f"{args.dataset}", "CityScapes")
    plot_IoU(results, f"{args.model}", f"{args.save_name}", f"{args.dataset}", "CityScapes")
    torch.save(model.state_dict(), f"{args.save_path}{args.save_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--dataset', type=str, required=True, choices=['cityscapes', 'gta5'], help='Dataset to train on')
    parser.add_argument('--model', type=str, required=True, choices=['bisenet18','bisenet101', 'deeplabv2'], help='Model to train')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--save_path', type=str, default='./models/', help='Path to save the model')
    parser.add_argument('--results_path', type=str, default='./results/', help='Path to save the results')
    parser.add_argument('--augmentation', default=False, action=argparse.BooleanOptionalAction, help='Use augmentations for GTA5?')
    parser.add_argument('--fda', type=float, default=None, help='Beta value for FDA')
    parser.add_argument('--dacs', default=False, action=argparse.BooleanOptionalAction, help='Use dacs?')
    parser.add_argument('--save_name', type=str, default='results', help='Name to save the results and model.')

    args = parser.parse_args()
    main(args)