import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import time
import numpy as np
from utils import get_id_to_label


def compute_flops(model:torch.nn.Module, height:int = 512, width:int = 1024)->dict:


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

def save_results(model:torch.nn.Module, model_results:list, filename:str, height:int, width:int, iterations:int, ignore_model_measurements:bool=False)->None:


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
            file.write(f"Training IoU for class {get_id_to_label()[i]} = {model_results[4][i]}\n")
        for i in range(0, 19):
            file.write(f"Validation IoU for class {get_id_to_label()[i]} = {model_results[5][i]}\n")
