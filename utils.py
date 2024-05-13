import numpy as np
from PIL import Image
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import time
def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
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


def fast_hist(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def label_to_rgb(label, height, width):
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
            rgb_image[i, j] = id_to_color.get(class_id, (0, 0, 0))  # Default to black if not found
    pil_image = Image.fromarray(rgb_image, 'RGB')
    return pil_image

def compute_flops(model, height = 256, width = 512):
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
    print(flop_count_table(flops))
    return flop_count_table(flops)

def get_latency_and_fps(model, height, width, iterations=1000):
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
    FPSs = []
    latencies = []
    model.eval()
    with torch.no_grad():
        for i in range(iterations):
            image = torch.zeros((1,3, height, width))
            start = time.time()
            model(image)
            end = time.time()
            latency = end-start
            latencies.append(latency)
            FPS = 1/latency
            FPSs.append(FPS)
    mean_latency = np.array(latencies).mean()*1000
    std_latency = np.array(latencies).std()*1000
    mean_FPS = np.array(FPSs).mean()
    std_FPS = np.array(FPSs).std()  
    return mean_latency, std_latency, mean_FPS, std_FPS

