# Real-time Unsupervised Domain Adaptation in Semantic Segmentation

## Overview

This project explores real-time domain adaptation techniques in semantic segmentation, a critical task in computer vision where each pixel of an image is classified into a specific category. Given the high cost and difficulty of obtaining labeled data for every possible domain, domain adaptation techniques become essential. We investigate the application of unsupervised domain adaptation (UDA) to real-time semantic segmentation networks, specifically evaluating how these networks perform when trained on synthetic data and tested on real-world data.

## Introduction

Semantic segmentation is a machine learning approach for labeling each pixel in an image with a specific class. This has applications in autonomous driving, medical diagnostics, robotics, and more. The project starts by evaluating DeepLabv2 and BiSeNet on the Cityscapes dataset. Then, BiSeNet's performance is tested by training it on synthetic GTA5 data and testing it on Cityscapes to highlight domain adaptation challenges. Advanced domain adaptation techniques like FDA and DACS are used to improve performance in real-time applications.

## Related Work

### Real-time Semantic Segmentation

Traditional methods in semantic segmentation, such as FCNs, have evolved to prioritize computational efficiency and accuracy. Networks like ENet, SegNet, and BiSeNet are notable for their balance between these factors, with BiSeNet introducing a dual-path framework for better efficiency and accuracy.

### Domain Adaptation in Semantic Segmentation

Domain adaptation addresses performance drops when applying a model trained on one domain to another with different data characteristics. Methods like adversarial training and input data manipulation, such as FDA and DACS, are explored to enhance generalization capabilities.

## Models

### DeepLabV2

DeepLabv2 uses atrous convolutions and ASPP to enhance segmentation performance, though it is computationally demanding.

### BiSeNet

BiSeNet combines spatial and contextual information pathways to achieve high efficiency and accuracy, making it suitable for real-time applications.

## Methods

### Domain Shift

Training models on synthetic datasets like GTA5 can mitigate the high cost of labeling real-world data. However, domain adaptation techniques are necessary to address performance degradation due to domain shift.

### FDA (Fourier Domain Adaptation)

FDA transfers low-frequency components from target images to source images to align domain appearances while preserving structural integrity.

### DACS (Domain Adaptation via Cross-Domain Mixed Sampling)

DACS generates mixed samples from source and target domains, enhancing the model's exposure to diverse data characteristics.

## Experiments

### Datasets

- **GTA5**: Synthetic images annotated for semantic segmentation.
- **Cityscapes**: Real-world images with high-quality annotations.

### Experimental Setup

- **Optimizer**: Adam with a learning rate of 2.5e-4 and a polynomial learning rate scheduler.
- **Batch Size**: 4
- **Training Epochs**: 50
- **Metrics**: Mean Intersection over Union (mIoU), latency, FPS, and FLOPs.

### Results

- BiSeNet showed higher efficiency for real-time applications compared to DeepLabV2.
- Data augmentation techniques improved performance, with FDA and DACS providing significant enhancements.
- DACS achieved the highest validation mIoU, demonstrating its effectiveness in real-time semantic segmentation.

## Conclusion

The study highlights the effectiveness of FDA and DACS in improving real-time semantic segmentation performance. These findings offer potential solutions for enhancing segmentation accuracy in practical applications, such as autonomous driving.

