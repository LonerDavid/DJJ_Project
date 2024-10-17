import cv2
import numpy as np
import math

import skimage as ski

file_path_1 = '/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/BABOON.BMP'
file_path_2 = '/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/peppers256.bmp'
file_path_3 = '/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/BABOON_BRIGHT.bmp'

image1 = cv2.imread(file_path_1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(file_path_2, cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread(file_path_3, cv2.IMREAD_GRAYSCALE)

nrmse13 = ski.metrics.normalized_root_mse(image1, image3)
psnr13 = ski.metrics.peak_signal_noise_ratio(image1, image3)
ssim13 = ski.metrics.structural_similarity(image1, image3)

nrmse12 = ski.metrics.normalized_root_mse(image1, image2)
psnr12 = ski.metrics.peak_signal_noise_ratio(image1, image2)
ssim12 = ski.metrics.structural_similarity(image1, image2)


print("Result of image 1&3 (skimage):")
print("NRMSE:", nrmse13)
print("PSNR:",psnr13)
print("SSIM", ssim13)
print("Result of image 1&2 (skimage):")
print("NRMSE:", nrmse12)
print("PSNR:",psnr12)
print("SSIM", ssim12)