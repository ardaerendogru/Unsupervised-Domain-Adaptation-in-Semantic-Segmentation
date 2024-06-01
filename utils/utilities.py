import numpy as np
from PIL import Image
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv
import PIL
def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Compute a fast histogram for evaluating segmentation metrics.

    This function calculates a 2D histogram where each entry (i, j) counts the number of pixels that have the true label i and the predicted label j with a mask.

    Args:
        a (np.ndarray): An array of true labels.
        b (np.ndarray): An array of predicted labels.
        n (int): The number of different labels.

    Returns:
        np.ndarray: A 2D histogram of size (n, n).
    """
    k = (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iou(hist: np.ndarray) -> np.ndarray:
    """
    Calculate the Intersection over Union (IoU) for each class.

    The IoU is computed for each class using the histogram of true and predicted labels. It is defined as the ratio of the diagonal elements of the histogram to the sum of the corresponding rows and columns, adjusted by the diagonal elements and a small epsilon to avoid division by zero.

    Args:
        hist (np.ndarray): A 2D histogram where each entry (i, j) is the count of pixels with true label i and predicted label j.

    Returns:
        np.ndarray: An array containing the IoU for each class.
    """
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def poly_lr_scheduler(optimizer:torch.optim.Optimizer, init_lr:float, iter:int, lr_decay_iter:int=1,
                      max_iter:int=50, power:float=0.9)->float:
    """
    Adjusts the learning rate of the optimizer for each iteration using a polynomial decay schedule.

    This function updates the learning rate of the optimizer based on the current iteration number and a polynomial decay schedule. The learning rate is calculated using the formula:
    
        lr = init_lr * (1 - iter/max_iter) ** power
    
    where `init_lr` is the initial learning rate, `iter` is the current iteration number, `max_iter` is the maximum number of iterations, and `power` is the exponent used in the polynomial decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.
        init_lr (float): The initial learning rate.
        iter (int): The current iteration number.
        lr_decay_iter (int): The iteration interval after which the learning rate is decayed. Default is 1.
        max_iter (int): The maximum number of iterations after which no more decay will happen.
        power (float): The exponent used in the polynomial decay of the learning rate.

    Returns:
        float: The updated learning rate.
    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    # lr = init_lr*(1 - iter/max_iter)**power
    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr

def label_to_rgb(label:np.ndarray, height:int, width:int)->PIL.Image:
    """
    Transforms a label matrix into a corresponding RGB image utilizing a predefined color map.

    This function maps each label identifier in a two-dimensional array to a specific color, thereby generating an RGB image. This is particularly useful for visualizing segmentation results where each label corresponds to a different segment class.

    Parameters:
        label (np.ndarray): A two-dimensional array where each element represents a label identifier.
        height (int): The desired height of the resulting RGB image.
        width (int): The desired width of the resulting RGB image.

    Returns:
        PIL.Image: An image object representing the RGB image constructed from the label matrix.
    """
    id_to_color = get_id_to_color()
    
    height, width = label.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            class_id = label[i, j]
            rgb_image[i, j] = id_to_color.get(class_id, (255, 255, 255))  # Default to white if not found
    pil_image = Image.fromarray(rgb_image, 'RGB')
    return pil_image

def generate_cow_mask(img_size:tuple, sigma:float, p:float, batch_size:int)->np.ndarray:

    """
    Generates a batch of cow masks based on a Gaussian noise model.

    Parameters:
        img_size (tuple): The size of the images (height, width).
        sigma (float): The standard deviation of the Gaussian filter applied to the noise.
        p (float): The desired proportion of the mask that should be 'cow'.
        batch_size (int): The number of masks to generate.

    Returns:
        np.ndarray: A batch of cow masks of shape (batch_size, 1, height, width).
    """
    N = np.random.normal(size=img_size) 
    Ns = gaussian_filter(N, sigma)
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    masks = []
    for i in range(batch_size):
        masks.append((Ns > t).astype(float).reshape(1,*img_size))
    return np.array(masks)

def get_id_to_label() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding labels.

    Returns:
        dict: A dictionary where keys are class IDs and values are labels.
    """
    return {
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

def get_id_to_color() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding colors.

    Returns:
        dict: A dictionary where keys are class IDs and values are RGB color tuples.
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
    }
    return id_to_color

def get_color_to_id() -> dict:
    """
    Returns a dictionary mapping RGB color tuples to their corresponding class IDs.

    Returns:
        dict: A dictionary where keys are RGB color tuples and values are class IDs.
    """
    id_to_color = get_id_to_color()
    color_to_id = {color: id for id, color in id_to_color.items()}
    return color_to_id

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N
