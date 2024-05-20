import numpy as np

def mutate_low_freq(source_amp:np.array, target_amp:np.array, beta:float)->np.array:
    """
    Mutates the low-frequency components of the source amplitude spectrum by replacing them with those from the target amplitude spectrum within a specified frequency band.

    Args:
        source_amp (np.array): The amplitude spectrum of the source image.
        target_amp (np.array): The amplitude spectrum of the target image.
        beta (float): The fraction of the radius used to define the low-frequency band.

    Returns:
        np.array: The mutated source amplitude spectrum with low-frequency components replaced by those of the target.
    """
    height, width = source_amp.shape[:2]
    center_y, center_x = height // 2, width // 2
    border = int(np.floor(min(height, width) * beta))
    y1, y2 = max(0, center_y-border), min(height-1, center_y+border)
    x1, x2 = max(0, center_x-border), min(width-1, center_x+border)
    source_amp[y1:y2,x1:x2,:] = target_amp[y1:y2,x1:x2,:]
    return source_amp

def FDA(source_image:np.array, target_img:np.array, beta:float)->np.array:
    """
    Performs Frequency Domain Adaptation (FDA) between a source image and a target image by swapping low-frequency components.

    Args:
        source_image (np.array): The source image from which high-frequency components are retained.
        target_img (np.array): The target image from which low-frequency components are taken.
        beta (float): The fraction of the radius used to define the low-frequency band.

    Returns:
        np.array: The adapted source image with low-frequency components from the target image.
    """
    source_image_ft = np.fft.fft2(source_image, axes=(0, 1))
    target_image_ft = np.fft.fft2(target_img, axes=(0, 1))
    source_image_ft_shift = np.fft.fftshift(source_image_ft,axes=(0, 1))
    target_image_ft_shift = np.fft.fftshift(target_image_ft,axes=(0, 1))
    source_amp, source_phase = np.abs(source_image_ft_shift), np.angle(source_image_ft_shift)
    target_amp, target_phase = np.abs(target_image_ft_shift), np.angle(target_image_ft_shift)
    source_image_ft_shift = mutate_low_freq(source_amp, target_amp, beta)
    source_image_ft_shift = np.fft.ifftshift(source_image_ft_shift * np.exp(1j * source_phase),axes=(0, 1))
    source_image_ft_shift = np.fft.ifft2(source_image_ft_shift, axes=(0, 1))
    source_image_ft_shift = np.real(source_image_ft_shift)
    return np.clip(source_image_ft_shift, 0, 255).astype(np.uint8)