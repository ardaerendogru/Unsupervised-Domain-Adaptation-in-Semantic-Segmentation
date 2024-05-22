import numpy as np
from PIL import Image
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import time
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

import matplotlib.pyplot as plt
id_to_label = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'light',
    7: 'sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    255: 'unlabeled'
}


def poly_lr_scheduler(optimizer:torch.optim.Optimizer, init_lr:float, iter:int, lr_decay_iter:int=1,
                      max_iter:int=300, power:float=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr


def fast_hist(a:np.ndarray, b:np.ndarray, n:int)->np.ndarray:
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist:np.ndarray)->np.ndarray:
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def label_to_rgb(label:np.ndarray, height:int, width:int):
    """
    Converts a label matrix to an RGB image using a predefined color map.

    Args:
        label (np.ndarray): The 2D array containing label ids.
        height (int): The height of the output image.
        width (int): The width of the output image.

    Returns:
        PIL.Image: The RGB image corresponding to the input label matrix.
    """
    id_to_color = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # light
    7: (220, 220, 0),     # sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
    19: (255,255,255)     #unlabeled
    }
    
    height, width = label.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            class_id = label[i, j]
            rgb_image[i, j] = id_to_color.get(class_id, (255, 255, 255))  # Default to white if not found
    pil_image = Image.fromarray(rgb_image, 'RGB')
    return pil_image

def compute_flops(model:torch.nn.Module, height:int = 512, width:int = 1024):
    """
    Computes the number of floating point operations (FLOPS) for a given model.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        height (int): The height of the input images.
        width (int): The width of the input images.
    Returns:
        str: A string containing the FLOPS count for the model.
    """

    image = torch.zeros((1,3, height, width))

    flops = FlopCountAnalysis(model.cpu(), image)
    table = flop_count_table(flops)
    ret_dict = {
        'Parameters':table.split('\n')[2].split('|')[2:][0].strip(),
        'FLOPS':table.split('\n')[2].split('|')[2:][1].strip(),
    }

    return ret_dict

def get_latency_and_fps(model: torch.nn.Module, height: int = 512, width: int = 1024, iterations: int = 1000) -> tuple:
    """
    Measures the latency and frames per second (FPS) of a given model on a specified input size over a number of iterations.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        height (int): The height of the input images.
        width (int): The width of the input images.
        iterations (int, optional): The number of iterations to measure. Defaults to 1000.

    Returns:
        tuple: A tuple containing the mean latency in milliseconds, the standard deviation of latency in milliseconds,
               the mean FPS, and the standard deviation of FPS.
    """
    latencies = []
    fps_records = []
    model.eval()
    
    with torch.no_grad():
        for _ in range(iterations):
            image = torch.zeros((1, 3, height, width))
            start_time = time.time()
            model(image)
            elapsed_time = time.time() - start_time
            latencies.append(elapsed_time)
            fps_records.append(1 / elapsed_time)
    
    mean_latency = np.mean(latencies) * 1000  # Convert to milliseconds
    std_latency = np.std(latencies) * 1000    # Convert to milliseconds
    mean_fps = np.mean(fps_records)
    std_fps = np.std(fps_records)
    
    return mean_latency, std_latency, mean_fps, std_fps

def save_results(model, model_results, filename, height, width, iterations, ignore_model_measurements=False):
    """
    Saves the model results and performance metrics to a specified file.

    Args:
        model (torch.nn.Module): The model whose results are being saved.
        model_results (list): A list containing the model's performance metrics and results.
        filename (str): The name of the file where the results will be saved.
        height (int): The height of the input images.
        width (int): The width of the input images.
        iterations (int): The number of iterations to measure latency and fps.
        ignore_model_measurements (bool, optional): If True, the model measurements will be ignored. Defaults to False.
    This function computes the FLOPS and parameters of the model using the `compute_flops` function,
    and measures the latency and FPS using the `get_latency_and_fps` function. It then writes these
    metrics, along with the training and validation IoU for each class, to a text file in the
    `./results/logs/` directory.
    """
    if not ignore_model_measurements:
        deeplab_params_flops = compute_flops(model, height=height, width=width)
        deeplab_latency_fps = get_latency_and_fps(model, height=height, width=width,iterations=iterations)

    with open(f'./results/logs/{filename}.txt', 'w') as file:
        if not ignore_model_measurements:
            file.write(f"Parameters : {deeplab_params_flops['Parameters']}\n")
            file.write(f"FLOPS : {deeplab_params_flops['FLOPS']}\n")
            file.write(f"Mean Latency = {deeplab_latency_fps[0]}\n")
            file.write(f"STD Latency = {deeplab_latency_fps[1]}\n")
            file.write(f"Mean FPS = {deeplab_latency_fps[2]}\n")
            file.write(f"STD FPS = {deeplab_latency_fps[3]}\n")
            file.write(f'Training Loss = {model_results[0][-1]}\n')
            file.write(f'Validation Loss = {model_results[1][-1]}\n')
            file.write(f'Training mIoU = {model_results[2][-1]}\n')
            file.write(f'Validation mIoU = {model_results[3][-1]}\n')
        for i in range(0, 19):
            file.write(f"Training IoU for class {id_to_label[i]} = {model_results[4][i]}\n")
        for i in range(0, 19):
            file.write(f"Validation IoU for class {id_to_label[i]} = {model_results[5][i]}\n")

def plot_loss(model_results, model_name, step, train_dataset, validation_dataset):
    """
    Plots the training and validation loss over epochs for a given model.

    Args:
        model_results (list): A list containing the model's loss values over epochs for both training and validation.
        model_name (str): The name of the model being evaluated.
        step (str): The current step or phase in the training/validation process.
        train_dataset (str): The name of the training dataset.
        validation_dataset (str): The name of the validation dataset.
    """
    epochs = range(len(model_results[0]))
    train_losses = model_results[0]
    validation_losses = model_results[1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f'Train vs. Validation Loss ({model_name})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(epochs, train_losses, label=f"Train Loss - {train_dataset}")
    ax.plot(epochs, validation_losses, label=f"Validation Loss - {validation_dataset}")
    ax.legend(loc="upper right")
    plt.show()
    fig.savefig(f"./results/images/{model_name}_{step}_loss.svg", format='svg', dpi=300)

def plot_mIoU(model_results, model_name, step, train_dataset, validation_dataset):
    """
    Plots the training and validation mean Intersection over Union (mIoU) over epochs for a given model.

    Args:
        model_results (list): A list containing the model's mIoU values over epochs for both training and validation.
        model_name (str): The name of the model being evaluated.
        step (str): The current step or phase in the training/validation process.
        train_dataset (str): The name of the training dataset.
        validation_dataset (str): The name of the validation dataset.
    """
    epochs = range(len(model_results[2]))
    train_mIoU = model_results[2]
    validation_mIoU = model_results[3]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f'Train vs. Validation mIoU ({model_name})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.plot(epochs, train_mIoU, label=f"Train mIoU - {train_dataset}")
    ax.plot(epochs, validation_mIoU, label=f"Validation mIoU - {validation_dataset}")
    ax.legend(loc="upper left")
    plt.show()
    fig.savefig(f"./results/images/{model_name}_{step}_mIoU.svg", format='svg', dpi=300)

def plot_IoU(model_results, model_name, step, train_dataset, validation_dataset):
    """
    Plots the training and validation Intersection over Union (IoU) for each class over epochs for a given model.

    Args:
        model_results (list): A list containing the model's IoU values over epochs for both training and validation.
        model_name (str): The name of the model being evaluated.
        step (str): The current step or phase in the training/validation process.
        train_dataset (str): The name of the training dataset.
        validation_dataset (str): The name of the validation dataset.
    """
    num_classes = 19
    class_names = [id_to_label[i] for i in range(num_classes)]
    train_iou = [model_results[4][i] for i in range(num_classes)]
    val_iou = [model_results[5][i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(num_classes)

    ax.bar(index, train_iou, bar_width, label=f'Train IoU - {train_dataset}')
    ax.bar(index + bar_width, val_iou, bar_width, label=f'Validation IoU - {validation_dataset}')

    ax.set_xlabel('Classes')
    ax.set_ylabel('IoU')
    ax.set_title(f'Training and Validation IoU for Each Class ({model_name})')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y')

    plt.tight_layout()
    plt.show()
    fig.savefig(f"./results/images/{model_name}_{step}_IoU_barplot.svg", format='svg', dpi=300)



def generate_cow_mask(img_size, sigma, p, batch_size):
    N = np.random.normal(size=img_size) 
    Ns = gaussian_filter(N, sigma)
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    masks = []
    for i in range(batch_size):
        masks.append((Ns > t).astype(float).reshape(1,*img_size))
    return np.array(masks)


