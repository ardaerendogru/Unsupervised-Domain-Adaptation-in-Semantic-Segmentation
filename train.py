from tqdm.notebook import tqdm
import torch
from utils import poly_lr_scheduler, fast_hist, per_class_iou
import numpy as np 
import wandb
import random
from utils import generate_cow_mask
from PIL import Image
import torch.nn.functional as F

def train(model, loss_fn, optimizer,train_dataloader,test_dataloader, epoch, project_name):
    """
    Train the model using the provided data loaders for a specified number of epochs.

    Args:
        model (torch.nn.Module): The neural network model to train.
        loss_fn (torch.nn.Module): The loss function used for optimization.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        train_dataloader (DataLoader): DataLoader for the training set.
        test_dataloader (DataLoader): DataLoader for the validation set.
        epoch (int): Number of epochs to train the model.
        project_name (str): Name of the project in Weights & Biases for logging.

    Returns:
        tuple: A tuple containing lists of training losses, validation losses, training mIoU scores, and validation mIoU scores.
    """
    train_losses = []
    validation_losses = []
    train_mious = []
    validation_mious = []

    init_lr = optimizer.param_groups[0]['lr']


    run = wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # Track hyperparameters and run metadata
        config={
            "init_lr": init_lr,
            "epochs": epoch,
        },
    )


    for i in tqdm(range(epoch)):
        model.train()
        train_loss = 0
        valid_loss = 0
        train_miou = 0
        validation_miou = 0


        if i == epoch-1:
            train_iou = np.zeros(19)
            validation_iou = np.zeros(19)

        for step, (image, label) in enumerate(train_dataloader):

            image, label = image.cuda(), label.type(torch.LongTensor).cuda()
            out = model(image)
            loss = loss_fn(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

            out_labels = torch.argmax(torch.softmax(out,dim=1),dim=1)
            hist = fast_hist(out_labels.cpu().numpy(),label.cpu().numpy(),20)
            train_miou += np.array(per_class_iou(hist)).flatten()[:19].mean()
            if i == epoch-1:
                train_iou += np.array(per_class_iou(hist)).flatten()[:19]


        lr = poly_lr_scheduler(optimizer, init_lr, i, max_iter=300, power=0.95)
        train_losses.append(train_loss/len(train_dataloader))
        train_mious.append(train_miou/len(train_dataloader))

        
        model.eval()
        with torch.inference_mode():
            for step, (image, label) in enumerate(test_dataloader):
                image, label = image.cuda(), label.type(torch.LongTensor).cuda()
                out = model(image)
                loss = loss_fn(out, label)
                valid_loss+=loss.item()
                
                out_labels = torch.argmax(torch.softmax(out,dim=1),dim=1)
                hist = fast_hist(out_labels.cpu().numpy(),label.cpu().numpy(),20)
                validation_miou += np.array(per_class_iou(hist)).flatten()[:19].mean()
                if i == epoch-1:
                    validation_iou += np.array(per_class_iou(hist)).flatten()[:19]

            validation_losses.append(valid_loss/len(test_dataloader))
            validation_mious.append(validation_miou/len(test_dataloader))

        if i == epoch-1:
            train_iou_avg = train_iou/len(train_dataloader)
            validation_iou_avg = validation_iou/len(test_dataloader)

        train_loss_avg = train_loss/len(train_dataloader)
        validation_loss_avg = valid_loss/len(test_dataloader)
        train_miou_avg = train_miou/len(train_dataloader)
        validation_miou_avg = validation_miou/len(test_dataloader)
        
        print(f'Epoch: {i}')
        print(f'Train Loss: {train_loss_avg}, Validation Loss: {validation_loss_avg}')
        print(f'Train mIoU: {train_miou_avg}, Validation mIoU: {validation_miou_avg}')
        wandb.log({"train_loss_avg": train_loss_avg, "validation_loss_avg": validation_loss_avg, "train_miou_avg": train_miou_avg, "validation_miou_avg": validation_miou_avg})

    return train_losses, validation_losses, train_mious, validation_mious, train_iou_avg, validation_iou_avg

def train_dacs(model, loss_fn, optimizer, source_dataloader, target_dataloader,test_dataloader, epoch, project_name, sigma):
    train_losses = []
    validation_losses = []
    train_mious = []
    validation_mious = []
    target_images = [image for image, _ in target_dataloader]
    init_lr = optimizer.param_groups[0]['lr']
    batch_size = source_dataloader.batch_size
    

    run = wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # Track hyperparameters and run metadata
        config={
            "init_lr": init_lr,
            "epochs": epoch,
        },
    )
    
    for i in tqdm(range(epoch)):
        model.train()
        train_loss = 0
        valid_loss = 0
        train_miou = 0
        validation_miou = 0
        

        if i == epoch-1:
            train_iou = np.zeros(19)
            validation_iou = np.zeros(19)

        for step, (image, label) in enumerate(source_dataloader):
            selected_target_images = random.choice(target_images)
            selected_target_images = F.interpolate(selected_target_images, size=(image.shape[-2],image.shape[-1]), mode='bilinear', align_corners=False)

            label_target = model(selected_target_images)
            label_target = torch.argmax(torch.softmax(label_target,dim=1),dim=1)
    
            mask = generate_cow_mask((image.shape[-2],image.shape[-1]),sigma,0.5, batch_size)
            mixed_images = image * mask + selected_target_images * (1-mask)
            mixed_labels = label * mask + label_target * (1-mask)

            image, label = image.cuda(), label.type(torch.LongTensor).cuda()
            mixed_images = mixed_images.cuda()
            mixed_labels = mixed_labels.cuda()
            out_source = model(image)
            loss_source = loss_fn(out_source, label)
            out_mixed = model(mixed_images)
            loss_mixed = loss_fn(out_mixed, mixed_labels)
            optimizer.zero_grad()
            loss = loss_source + loss_mixed
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

            out_source_labels = torch.argmax(torch.softmax(out_source,dim=1),dim=1)
            hist = fast_hist(out_source_labels.cpu().numpy(),label.cpu().numpy(),20)
            train_miou += np.array(per_class_iou(hist)).flatten()[:19].mean()/2
            if i == epoch-1:
                train_iou += np.array(per_class_iou(hist)).flatten()[:19]/2

            out_mixed_labels = torch.argmax(torch.softmax(out_mixed,dim=1),dim=1)
            hist = fast_hist(out_mixed_labels.cpu().numpy(),label.cpu().numpy(),20)
            train_miou += np.array(per_class_iou(hist)).flatten()[:19].mean()/2
            if i == epoch-1:
                train_iou += np.array(per_class_iou(hist)).flatten()[:19]/2


        lr = poly_lr_scheduler(optimizer, init_lr, i, max_iter=300, power=0.95)
        train_losses.append(train_loss/len(source_dataloader))
        train_mious.append(train_miou/len(source_dataloader))

        
        model.eval()
        with torch.inference_mode():
            for step, (image, label) in enumerate(test_dataloader):
                image, label = image.cuda(), label.type(torch.LongTensor).cuda()
                out = model(image)
                loss = loss_fn(out, label)
                valid_loss+=loss.item()
                
                out_labels = torch.argmax(torch.softmax(out,dim=1),dim=1)
                hist = fast_hist(out_labels.cpu().numpy(),label.cpu().numpy(),20)
                validation_miou += np.array(per_class_iou(hist)).flatten()[:19].mean()
                if i == epoch-1:
                    validation_iou += np.array(per_class_iou(hist)).flatten()[:19]

            validation_losses.append(valid_loss/len(test_dataloader))
            validation_mious.append(validation_miou/len(test_dataloader))

        if i == epoch-1:
            train_iou_avg = train_iou/len(source_dataloader)
            validation_iou_avg = validation_iou/len(test_dataloader)

        train_loss_avg = train_loss/len(source_dataloader)
        validation_loss_avg = valid_loss/len(test_dataloader)
        train_miou_avg = train_miou/len(source_dataloader)
        validation_miou_avg = validation_miou/len(test_dataloader)
        
        print(f'Epoch: {i}')
        print(f'Train Loss: {train_loss_avg}, Validation Loss: {validation_loss_avg}')
        print(f'Train mIoU: {train_miou_avg}, Validation mIoU: {validation_miou_avg}')
        wandb.log({"train_loss_avg": train_loss_avg, "validation_loss_avg": validation_loss_avg, "train_miou_avg": train_miou_avg, "validation_miou_avg": validation_miou_avg})

    return train_losses, validation_losses, train_mious, validation_mious, train_iou_avg, validation_iou_avg
