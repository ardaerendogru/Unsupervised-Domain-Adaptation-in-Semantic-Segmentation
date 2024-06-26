from tqdm.notebook import tqdm
import torch
from utils import poly_lr_scheduler, fast_hist, per_class_iou
import numpy as np 
import random
from utils import generate_cow_mask, generate_class_mask, mix
import torch.nn.functional as F
from typing import Tuple, List
from typing import Optional
import time
def train_step(
        model:torch.nn.Module, 
        dataloader:torch.utils.data.DataLoader, 
        optimizer:torch.optim.Optimizer, 
        loss_fn:torch.nn.Module, 
        device:torch.device, 
        model_name:str,
        class_number:int =  19
        ) -> Tuple[float, float, float]:
    
    """
    Perform a training step for a given model using the provided dataloader.

    This function processes a batch of images and labels from the dataloader, computes the loss using the specified loss function, and updates the model parameters using the optimizer. It supports different model architectures by checking the `model_name`. For the 'bisenet' model, it handles multiple outputs and computes the loss accordingly. The function also calculates the mean Intersection over Union (mIoU) for performance evaluation.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the dataset.
        optimizer (torch.optim.Optimizer): The optimizer to update model's weights.
        loss_fn (torch.nn.Module): The loss function to measure the model's performance.
        device (torch.device): The device tensors will be moved to before computation.
        model_name (str): The name of the model architecture being used.
        class_number (int, optional): The number of classes in the dataset. Defaults to 19.

    Returns:
        Tuple[float, float, np.ndarray]: A tuple containing the average loss, average mIoU, and average per-class IoU across the dataloader.
    """

    model.train()
    total_loss = 0
    total_miou = 0
    total_iou = np.zeros(class_number)
    for image, label in dataloader: 
        image, label = image.to(device), label.type(torch.LongTensor).to(device)
        if model_name == "bisenet":
            out, sup1, sup2 = model(image)
            loss = loss_fn(out, label) #+ (loss_fn(sup1, label) + loss_fn(sup2, label))*0
        else:
            out = model(image)
            loss = loss_fn(out, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        out_labels = torch.argmax(torch.softmax(out, dim=1), dim=1)
        hist = fast_hist(out_labels.cpu().numpy(), label.cpu().numpy(), class_number)
        pc_iou = np.array(per_class_iou(hist)).flatten()
        total_iou += pc_iou
        total_miou +=pc_iou.sum()
    
    avg_loss = total_loss / len(dataloader)
    avg_miou = total_miou / (len(dataloader) * class_number)
    avg_iou = total_iou / len(dataloader)
    return avg_loss, avg_miou, avg_iou


def train_step_dacs(
        model:torch.nn.Module, 
        loss_fn:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        dataloader:torch.utils.data.DataLoader, 
        target_images:List[torch.Tensor],
        sigma:float,
        class_number:int = 19,
        device: torch.device = torch.device('cuda'),
        model_name: str = 'bisenet'
        ) -> Tuple[float, float, float]:


    """
    Perform a training step using Domain Adaptive Class Mixing (DACM) strategy on a given model.

    This function takes a model, loss function, optimizer, a dataloader for source domain, and a list of target domain images.
    It performs the training by mixing source domain images with target domain images using a generated mask, and then
    computes the loss and updates the model parameters accordingly. The function also calculates the mean Intersection
    over Union (mIoU) for both source and mixed images.

    Args:
        model (torch.nn.Module): The model to be trained.
        loss_fn (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing source domain data.
        target_images (List[torch.Tensor]): A list of tensors containing target domain images.
        sigma (float): The standard deviation used in the Gaussian distribution for mask generation.
        class_number (int, optional): The number of classes in the dataset. Defaults to 19.
        device (torch.device): The device tensors will be moved to before computation.
        model_name (str): The name of the model architecture being used.

    Returns:
        Tuple[float, float, float]: A tuple containing the average loss, average mIoU for source images,
        and average mIoU for mixed images.
    """


    total_loss = 0
    total_source_miou = 0
    total_mixed_miou = 0
    total_source_iou = np.zeros(class_number)
    total_mixed_iou = np.zeros(class_number)

    model.train()
    for step, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.long().to(device)
        selected_target_images = random.choice(target_images).to(device)
        selected_target_images = selected_target_images[0:image.shape[0],:,:,:]
        selected_target_images = F.interpolate(selected_target_images, size=(image.shape[-2], image.shape[-1]), mode='bilinear', align_corners=False)   
        
        label_target, _, _ = model(selected_target_images)
        label_target = torch.argmax(torch.softmax(label_target, dim=1), dim=1).long()
        masks = []
        for i in range(image.size(0)):  # Iterate over each image in the batch
            image_classes = torch.unique(label[i]).sort()[0]
            image_classes = image_classes[image_classes != 255]  # Exclude the ignore class
            nclasses = image_classes.shape[0]
          
            
            selected_classes = image_classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()].cuda()
            masks.append(generate_class_mask(label[i], selected_classes))

        mask = torch.stack(masks)

        mixed_images = image * mask.unsqueeze(1) + selected_target_images * (1 - mask.unsqueeze(1))
        mixed_labels = label * mask + label_target * (1 - mask)

        mixed_images = mixed_images.float()
        mixed_labels = mixed_labels.long()
        if model_name == "bisenet":
            out_source, aux1, aux2 = model(image)
            loss_source = loss_fn(out_source, label) #+ loss_fn(aux1, label) + loss_fn(aux2, label)
        else:
            out_source = model(image)
            loss_source = loss_fn(out_source, label)
        total_loss+=loss_source.item()
        
        if model_name == "bisenet":
            out_mixed, aux1mix, aux2mix = model(mixed_images)
            loss_mixed = loss_fn(out_mixed, mixed_labels) #+ loss_fn(aux1mix, mixed_labels) + loss_fn(aux2mix, mixed_labels)
        else:
            out_mixed = model(mixed_images)
            loss_mixed = loss_fn(out_mixed, mixed_labels)
        total_loss+=loss_mixed.item()

        optimizer.zero_grad()
        loss = loss_source + loss_mixed
        loss.backward()
        optimizer.step()

        out_source_labels = torch.argmax(torch.softmax(out_source,dim=1),dim=1)
        hist_source = fast_hist(out_source_labels.cpu().numpy(),label.cpu().numpy(),class_number)
        out_source_per_class_iou = np.array(per_class_iou(hist_source)).flatten()
        total_source_iou += out_source_per_class_iou
        total_source_miou += out_source_per_class_iou.sum()


        out_mixed_labels = torch.argmax(torch.softmax(out_mixed,dim=1),dim=1)
        hist_mixed = fast_hist(out_mixed_labels.cpu().numpy(),label.cpu().numpy(),class_number)
        out_mixed_per_class_iou = np.array(per_class_iou(hist_mixed)).flatten()
        total_mixed_iou += out_mixed_per_class_iou
        total_mixed_miou += out_mixed_per_class_iou.sum()

    avg_loss = total_loss/(len(dataloader)*2)
    avg_miou = (total_source_miou + total_mixed_miou)/(len(dataloader)*2*class_number)
    avg_iou = (total_source_iou + total_mixed_iou)/(len(dataloader)*2)
    return avg_loss, avg_miou, avg_iou



def validation_step(
        model:torch.nn.Module, 
        dataloader:torch.utils.data.DataLoader, 
        loss_fn:torch.nn.Module, 
        device:torch.device,
        class_number:int = 19,
        ) -> Tuple[float, float, float]:
    
    """
    Perform a validation step to evaluate the model on a given dataset.

    This function evaluates the model's performance on a validation dataset using the provided loss function.
    It computes the average loss, mean intersection over union (mIoU), and per-class IoU for the dataset.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the validation dataset.
        loss_fn (torch.nn.Module): The loss function used to evaluate the model's performance.
        device (torch.device): The device tensors will be sent to for model computation.
        class_number (int, optional): The number of classes in the dataset. Defaults to 19.

    Returns:
        Tuple[float, float, np.ndarray]: A tuple containing the average loss, average mIoU, and per-class IoU across the validation dataset.
    """
    model.eval()
    total_loss = 0
    total_miou = 0
    total_iou = np.zeros(class_number)
    with torch.inference_mode():
        for image, label in dataloader:
            image, label = image.to(device), label.type(torch.LongTensor).to(device)
            out = model(image)
            loss = loss_fn(out, label)
            total_loss += loss.item()
            out_labels = torch.argmax(torch.softmax(out, dim=1), dim=1)
            hist = fast_hist(out_labels.cpu().numpy(), label.cpu().numpy(), class_number)
            pc_iou = np.array(per_class_iou(hist)).flatten()
            total_iou += pc_iou
            total_miou += pc_iou.sum()
    
    avg_loss = total_loss / len(dataloader)
    avg_miou = total_miou / (len(dataloader) * class_number)
    avg_iou = total_iou / len(dataloader)
    return avg_loss, avg_miou, avg_iou

def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        epochs: int,
        device: torch.device,
        model_name: str = 'bisenet',
        target_dataloader: Optional[torch.utils.data.DataLoader] = None,
        class_number: int = 19,
        sigma :int = 150,
        power:float = 0.9
        ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    
    """
    Trains the model over a specified number of epochs, optionally using domain adaptation for semi-supervised learning.

    This function orchestrates the training loop, which includes both training and validation phases per epoch. It supports
    domain adaptation by utilizing unlabeled target data if provided. The function logs and returns the training and validation
    losses and mean intersection over union (mIoU) metrics.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        loss_fn (torch.nn.Module): The loss function used for training.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        validation_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        epochs (int): Total number of epochs to train the model.
        device (torch.device): The device to perform computations on.
        model_name (str, optional): Name of the model, used for logging purposes. Defaults to 'bisenet'.
        target_dataloader (Optional[torch.utils.data.DataLoader]): DataLoader for target domain data, used in domain adaptation. Defaults to None.
        class_number (int, optional): Number of classes in the dataset. Defaults to 19.
        sigma (int, optional): Hyperparameter for domain adaptation, controlling the sharpness of the label distribution. Defaults to 150.
        power (float, optional): Exponent for polynomial learning rate decay. Defaults to 0.9.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
            - List of training losses per epoch.
            - List of validation losses per epoch.
            - List of training mIoUs per epoch.
            - List of validation mIoUs per epoch.
            - List of training per-class IoUs per epoch (if applicable).
            - List of validation per-class IoUs per epoch (if applicable).
    """
    train_losses = []
    validation_losses = []
    train_mious = []
    validation_mious = []
    init_lr = optimizer.param_groups[0]['lr']
    if target_dataloader:
        target_images = [image for image, _ in target_dataloader]
    if 'bisenet' in model_name:
        model_name = 'bisenet'
    elif 'deeplab' in model_name:
        model_name = 'deeplab'
    for epoch in tqdm(range(epochs)):
        if target_dataloader:
            train_loss, train_miou, train_iou = train_step_dacs(model= model,
                                                                loss_fn= loss_fn,
                                                                optimizer= optimizer,
                                                                dataloader= train_dataloader,
                                                                target_images= target_images,
                                                                sigma= sigma,
                                                                class_number= class_number,
                                                                device= device,
                                                                model_name= model_name)
            
        else:
            train_loss, train_miou, train_iou = train_step(model= model,
                                                           dataloader= train_dataloader,
                                                           optimizer= optimizer,
                                                           loss_fn= loss_fn,
                                                           device= device,
                                                           model_name= model_name,
                                                           class_number= class_number)
            

        validation_loss, validation_miou, validation_iou = validation_step(model= model,
                                                                           dataloader= validation_dataloader,
                                                                           loss_fn= loss_fn,
                                                                           device= device,
                                                                           class_number= class_number)
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_mious.append(train_miou)
        validation_mious.append(validation_miou)
        
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss}, Validation Loss: {validation_loss}')
        print(f'Train mIoU: {train_miou}, Validation mIoU: {validation_miou}')

        poly_lr_scheduler(optimizer = optimizer,
                          init_lr = init_lr,
                          iter = epoch, 
                          max_iter = epochs,
                          power = power)

    return train_losses, validation_losses, train_mious, validation_mious, train_iou, validation_iou

        