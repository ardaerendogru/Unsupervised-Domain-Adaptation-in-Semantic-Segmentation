from tqdm.notebook import tqdm
import torch
from utils import poly_lr_scheduler, fast_hist, per_class_iou
import numpy as np 
import wandb
wandb.login()

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

